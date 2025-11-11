import importlib, inspect, types, math, random, os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils as vutils
from torchvision.transforms import InterpolationMode as IM

# -----------------------
# Quick config
# -----------------------
model_mode   = "unet"
dataset_mode = "drive"
loss_mode    = "focal"            # uses your lib.losses.FocalLoss
data_root    = Path("/dtu/datasets1/02516/PH2_Dataset_images") # DRIVE or PH2_Dataset_images
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_tfm = T.Compose([
    T.Resize((512, 512), interpolation=IM.BILINEAR, antialias=True),
    T.ToTensor(),
])
def mask_tfm(msk):
    # Resize the mask using nearest-neighbor interpolation to preserve labels
    m = T.Resize((512, 512), interpolation=IM.NEAREST)(msk)
    # Convert PIL → tensor (1,H,W) in [0,1]
    m = T.ToTensor()(m)
    # Threshold at 0.5 to get binary mask, then remove the channel dimension
    return (m >= 0.5).to(torch.long).squeeze(0)

# -----------------------
# Repro
# -----------------------
def set_seed(s: int = 1337):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed(1337)

# -----------------------
# Safe imports (your factory helpers)
# -----------------------
def import_module(primary: str, *fallbacks: str):
    try:
        return importlib.import_module(primary)
    except Exception:
        for fb in fallbacks:
            try:
                return importlib.import_module(fb)
            except Exception:
                continue
        raise

def import_class(primary: str, fallback: str, attr: str):
    def _load(mn): return importlib.import_module(mn)
    try:
        mod = _load(primary)
    except Exception:
        mod = _load(fallback)
    obj = getattr(mod, attr, None)
    if obj is None:
        for base in (primary, fallback):
            try:
                sub = _load(f"{base}.{attr}")
                if hasattr(sub, attr) and inspect.isclass(getattr(sub, attr)):
                    return getattr(sub, attr)
                for _, member in inspect.getmembers(sub, inspect.isclass):
                    if member.__module__ == sub.__name__:
                        return member
            except Exception:
                pass
        raise ImportError(f"Could not find class '{attr}' in {primary} or {fallback}.")
    if isinstance(obj, types.ModuleType):
        if hasattr(obj, attr) and inspect.isclass(getattr(obj, attr)):
            return getattr(obj, attr)
        if hasattr(obj, "Model") and inspect.isclass(getattr(obj, "Model")):
            return getattr(obj, "Model")
        for _, member in inspect.getmembers(obj, inspect.isclass):
            if member.__module__ == obj.__name__:
                return member
        raise ImportError(f"Module '{obj.__name__}' does not expose a model class.")
    if not inspect.isclass(obj):
        raise TypeError(f"'{attr}' in {mod.__name__} is not a class.")
    return obj

# -----------------------
# Factories (use your existing files)
# -----------------------
MODEL_MAP = {
    "unet": ("UNetModel",),
    "dilated": ("DilatedNetModel",),
    "encdec": ("EncDecModel",),
}

DatasetMap = {
    "drive": ("DRIVE_dataset", str(data_root)),
    "ph2": ("PH2_dataset", str(data_root)),
    "click": ("Clickpoints_dataset", str(data_root)),  # uses PH2 images, generates clicks
}

# Select model + dataset
ModelClass = import_class("Project3.lib.model", "lib.model", MODEL_MAP[model_mode][0])
model = ModelClass().to(device)

DatasetClassName, dataset_root_dir = DatasetMap[dataset_mode]
DatasetClass = import_class("Project3.lib.dataset", "lib.dataset", DatasetClassName)

# Define transforms
if dataset_mode == "click":
    # Assuming ClickDataset returns images + click masks
    img_tfm = T.Compose([
        T.Resize((512, 512), interpolation=IM.BILINEAR, antialias=True),
        T.ToTensor(),
        # Optional: normalize clicks if stored as grayscale heatmaps
        # T.Normalize(mean=[0.5], std=[0.5]),
    ])

    def mask_tfm(msk):
        # Click maps are discrete, so keep nearest-neighbor
        m = T.Resize((512, 512), interpolation=IM.NEAREST)(msk)
        m = T.ToTensor()(m)
        return (m >= 0.5).to(torch.long).squeeze(0)

else:
    # Fallback for DRIVE / PH2
    img_tfm = T.Compose([
        T.Resize((512, 512), interpolation=IM.BILINEAR, antialias=True),
        T.ToTensor(),
    ])
    def mask_tfm(msk):
        m = T.Resize((512, 512), interpolation=IM.NEAREST)(msk)
        m = T.ToTensor()(m)
        return (m >= 0.5).to(torch.long).squeeze(0)

# Initialize dataset loaders
train_ds = DatasetClass(root_dir=dataset_root_dir, split="train",
                        image_transform=img_tfm, mask_transform=mask_tfm)
val_ds   = DatasetClass(root_dir=dataset_root_dir, split="val",
                        image_transform=img_tfm, mask_transform=mask_tfm)

print(f"Train dataset length: {len(train_ds)}")
print(f"Validation dataset length: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)


# -----------------------
# Label stats -> α for FocalLoss (and optional bias init)
# -----------------------
_, lbls = next(iter(val_loader))
counts = torch.bincount(lbls.view(-1), minlength=2)
pos = counts[1].item()
neg = counts[0].item()
pos_frac = pos / max(pos + neg, 1)
alpha = float(1.0 - pos_frac)       # focus rare positives
print(f"[stats] pos_frac≈{pos_frac:.4f}  α={alpha:.3f}  counts={counts.tolist()}")

# Try to initialize a 1-logit head bias from prior (if such a layer exists)
def try_init_last_single_logit_bias(m, prior: float):
    last_1ch = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d) and mod.out_channels == 1:
            last_1ch = mod
    if last_1ch is not None and last_1ch.bias is not None:
        b = math.log(max(prior, 1e-6) / max(1 - prior, 1e-6))
        with torch.no_grad():
            last_1ch.bias.fill_(b)
        print(f"[init] Set last 1-logit conv bias to prior={prior:.4f} (b={b:.3f})")
try_init_last_single_logit_bias(model, pos_frac)

# -----------------------
# Losses & measures
# -----------------------
losses   = import_module("Project3.lib.losses", "lib.losses")
measures = import_module("measure", "Project3.lib.measure", "lib.measure")

def make_criterion(mode: str) -> nn.Module:
    m = mode.lower()
    if m == "focal":
        return losses.FocalLoss(alpha=alpha, gamma=2.0, reduction="mean")
    if m in {"ce", "crossentropy"}:
        # Weights to counter class imbalance (for 2-class heads only)
        w = torch.tensor([1.0, max(1.0, neg/max(pos,1))], device=device, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=w)
    if m == "dice":
        return losses.DiceLoss()
    raise ValueError(f"Unknown loss: {mode}")

criterion: nn.Module = make_criterion(loss_mode)

# -----------------------
# Helpers (C=1 or C=2 safe)
# -----------------------
def _probs_preds_from_logits(logits: torch.Tensor, threshold: float = 0.5):
    """
    Accepts logits (N,C,H,W) with C in {1,2}. Returns:
      probs: (N,H,W) in [0,1]
      preds: (N,H,W) in {0,1}
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tuple(logits.shape)}")
    C = logits.size(1)
    if C == 1:
        probs = torch.sigmoid(logits[:, 0])
    elif C == 2:
        probs = torch.softmax(logits, dim=1)[:, 1]
    else:
        raise ValueError(f"Unsupported head channels C={C}")
    preds = (probs >= threshold).float()
    return probs, preds

def _to_3ch(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3: t = t.unsqueeze(1)
    if t.size(1) == 1: t = t.repeat(1,3,1,1)
    return t

def save_visuals(inputs, labels, probs, preds, out_dir, epoch, max_items=4):
    os.makedirs(out_dir, exist_ok=True)
    k = min(max_items, inputs.size(0))
    x  = inputs[:k].detach().cpu()
    y  = labels[:k].detach().cpu()
    pr = probs[:k].detach().cpu()
    pd = preds[:k].detach().cpu()
    if y.dim() == 3:  y  = y.unsqueeze(1)
    if pr.dim() == 3: pr = pr.unsqueeze(1)
    if pd.dim() == 3: pd = pd.unsqueeze(1)
    grid = vutils.make_grid(torch.cat([_to_3ch(x), _to_3ch(y), _to_3ch(pr), _to_3ch(pd)], dim=0),
                            nrow=k, padding=2, normalize=False)
    vutils.save_image(grid, os.path.join(out_dir, f"epoch_{epoch:03d}.png"))

def current_lr(optim: torch.optim.Optimizer) -> float:
    return optim.param_groups[0]['lr']

def _probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits (N,C,H,W) with C in {1,2} into positive-class probabilities (N,H,W) in [0,1].
    """
    if logits.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tuple(logits.shape)}")
    C = logits.size(1)
    if C == 1:
        return torch.sigmoid(logits[:, 0])
    elif C == 2:
        return torch.softmax(logits, dim=1)[:, 1]
    else:
        raise ValueError(f"Unsupported head channels C={C}")


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None, threshold: float = 0.35):
    """
    Threshold-free validation:
      Dice_soft = (2*TP) / (sum(p) + sum(y))
      IoU_soft  = TP / (TP + FP + FN)
      Acc_soft  = (TP + TN) / (TP + TN + FP + FN)
      Sens_soft = TP / (TP + FN)
      Spec_soft = TN / (TN + FP)
    where TP = sum(p*y), FP = sum(p*(1-y)), TN = sum((1-p)*(1-y)), FN = sum((1-p)*y)
    The 'threshold' arg is ignored for metrics (kept only for API compatibility).
    """
    model.eval()
    running_loss, num_samples = 0.0, 0

    tp = fp = tn = fn = 0.0
    sum_p = sum_y = 0.0
    eps = 1e-8

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()  # (N,H,W) ∈ {0,1}

        logits = model(inputs)  # (N,C,H,W)

        if criterion is not None:
            # If your criterion expects long targets for 2-class heads, handle it:
            if logits.size(1) == 2:
                loss = criterion(logits, labels.long())
            else:
                loss = criterion(logits, labels)
            running_loss += float(loss.item()) * inputs.size(0)
            num_samples += inputs.size(0)

        probs = _probs_from_logits(logits)  # (N,H,W) in [0,1]

        tp += (probs * labels).sum().item()
        fp += (probs * (1.0 - labels)).sum().item()
        tn += ((1.0 - probs) * (1.0 - labels)).sum().item()
        fn += ((1.0 - probs) * labels).sum().item()

        sum_p += probs.sum().item()
        sum_y += labels.sum().item()

    dice = (2.0 * tp + eps) / (sum_p + sum_y + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    acc  = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    sens = (tp + eps) / (tp + fn + eps)
    spec = (tn + eps) / (tn + fp + eps)

    avg_loss = (running_loss / max(num_samples, 1)) if criterion is not None else 0.0
    return avg_loss, float(dice), float(iou), float(acc), float(sens), float(spec)

# -----------------------
# Training
# -----------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    checkpoint_dir: str = "checkpoints",
    run_name: str = "run",
    early_stopping_patience: Optional[int] = 20,
    threshold: float = 0.5,
    vis_every: int = 1,
    vis_max_items: int = 4,
    vis_dir: Optional[str] = None,
):
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_dice = -float("inf"); epochs_no_improve = 0
    history = {k: [] for k in ["train_loss","val_loss","val_dice","val_iou","val_acc","val_sens","val_spec","lr"]}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, num_samples = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)         # (N,C,H,W)
                loss = criterion(logits, labels)  # focal handles both heads

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = inputs.size(0)
            running_loss += loss.item() * bs * grad_accum_steps
            num_samples += bs

        scheduler.step()

        # Validation
        val_loss, val_dice, val_iou, val_acc, val_sens, val_spec = evaluate(
            model, val_loader, device, criterion=criterion, threshold=threshold
        )

        # Visualizations
        vis_out_dir = vis_dir or os.path.join(checkpoint_dir, "vis", run_name)
        if (epoch % max(vis_every, 1)) == 0:
            model.eval()
            with torch.inference_mode():
                for vis_inputs, vis_labels in val_loader:
                    vis_inputs = vis_inputs.to(device, non_blocking=True)
                    vis_labels = vis_labels.to(device, non_blocking=True)
                    logits_vis = model(vis_inputs)
                    probs_vis, preds_vis = _probs_preds_from_logits(logits_vis, threshold)
                    save_visuals(vis_inputs, vis_labels, probs_vis, preds_vis, vis_out_dir, epoch, max_items=vis_max_items)
                    break  # only first batch

        # Logging
        train_loss_epoch = running_loss / max(num_samples, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_acc"].append(val_acc)
        history["val_sens"].append(val_sens)
        history["val_spec"].append(val_spec)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.3e} | "
              f"train_loss={train_loss_epoch:.4f} | val_loss={val_loss:.4f} | "
              f"Dice={val_dice:.4f} | IoU={val_iou:.4f} | "
              f"Acc={val_acc:.4f} | Sens={val_sens:.4f} | Spec={val_spec:.4f}")

        # Save last
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
        }, os.path.join(checkpoint_dir, f"{run_name}_last.pt"))

        # Save best (by Dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice; epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
            }, os.path.join(checkpoint_dir, f"{run_name}_best.pt"))
            print(f"  ↳ New best Dice: {best_val_dice:.4f} (checkpoint saved)")
        else:
            epochs_no_improve += 1
            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping (no improvement in {early_stopping_patience} epochs).")
                break

    return history

# -----------------------
# Run
# -----------------------
history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    criterion=criterion,
    epochs=200,
    lr=1e-4,
    weight_decay=1e-4,
    checkpoint_dir="checkpoints",
    run_name=f"{model_mode}_{dataset_mode}_{loss_mode}",
    early_stopping_patience=200,
    vis_every=10,
    vis_max_items=4,
    vis_dir=None,
)