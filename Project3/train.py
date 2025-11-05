import argparse
import importlib
import inspect
import math
import types
from pathlib import Path
import os
import json
from datetime import datetime
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils as vutils
from torchvision.transforms import InterpolationMode

# -----------------------
# Config (quick flags)
# -----------------------
model_mode   = "unet"
dataset_mode = "drive"
loss_mode    = "bce"

data_root = Path("/dtu/datasets1/02516/DRIVE")

img_tfm = T.Compose([
    T.Resize((128, 128), interpolation=InterpolationMode.BILINEAR, antialias=True),
    T.ToTensor(),
])
def mask_tfm(msk):
    m = T.Resize((128, 128), interpolation=InterpolationMode.NEAREST)(msk)
    m = T.ToTensor()(m)
    return (m >= 0.5).float()

img_resize = T.Resize((512, 512), interpolation=InterpolationMode.BILINEAR, antialias=True)
msk_resize = T.Resize((512, 512), interpolation=InterpolationMode.NEAREST)

tfm_train = T.Compose([T.ToTensor(), T.Resize((128, 128), antialias=True)])
tfm_val   = T.Compose([T.ToTensor(), T.Resize((128, 128), antialias=True)])



# -----------------------
# Repro & device
# -----------------------
def set_seed(s: int = 1337):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(1337)

# -----------------------
# Resilient import helpers
# -----------------------
def import_module(primary: str, *fallbacks: str):
    """Try `primary`, else each fallback in order."""
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
    """
    Try to get class `attr` from module `primary` (or `fallback`).
    If it's a submodule, look inside it for a same-named class (or a sensible default).
    """
    def _load(modname):
        return importlib.import_module(modname)

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
                for name, member in inspect.getmembers(sub, inspect.isclass):
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
        for name, member in inspect.getmembers(obj, inspect.isclass):
            if member.__module__ == obj.__name__:
                return member
        raise ImportError(f"Module '{obj.__name__}' does not expose a model class.")

    if not inspect.isclass(obj):
        raise TypeError(f"'{attr}' in {mod.__name__} is not a class (got {type(obj)}).")
    return obj

# -----------------------
# Model factory
# -----------------------
MODEL_MAP = {
    "unet":   ("UNetModel",),
    "dilated":("DilatedNetModel",),
    "encdec": ("EncDecModel",),
}
if model_mode not in MODEL_MAP:
    raise ValueError(f"Unknown model: {model_mode}")

ModelClass = import_class("Project3.lib.model", "lib.model", MODEL_MAP[model_mode][0])
model = ModelClass().to(device)

# -----------------------
# Dataset factory
# -----------------------
DATASET_MAP = {
    "drive": ("DRIVE_dataset", str(data_root)),
    "ph2":   ("PH2_dataset", "data/PH2"),
}
if dataset_mode not in DATASET_MAP:
    raise ValueError(f"Unknown dataset: {dataset_mode}")

DatasetClassName, dataset_root_dir = DATASET_MAP[dataset_mode]
DatasetClass = import_class("Project3.lib.dataset", "lib.dataset", DatasetClassName)

train_ds = DatasetClass(root_dir=dataset_root_dir, split='train', image_transform=img_tfm, mask_transform=mask_tfm)
val_ds   = DatasetClass(root_dir=dataset_root_dir, split='val', image_transform=img_tfm, mask_transform=mask_tfm)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

imgs, lbls = next(iter(val_loader))
print("val batch labels:", lbls.min().item(), lbls.max().item(), lbls.mean().item())

# -----------------------
# Losses & Measures (local measure.py preferred)
# -----------------------
losses   = import_module("Project3.lib.losses", "lib.losses")
measures = import_module("measure", "Project3.lib.measure", "lib.measure")

def make_criterion(mode: str) -> nn.Module:
    mode = mode.lower()
    if mode == "bce":
        return nn.CrossEntropyLoss(reduction="mean")
    if mode == "dice":
        return losses.DiceLoss()
    if mode == "focal":
        return losses.FocalLoss()
    if mode in {"bce_tv", "bce-total-variation", "bce_tv_reg"}:
        return losses.BCELoss_TotalVariation()
    raise ValueError(f"Unknown loss: {mode}")

criterion: nn.Module = make_criterion(loss_mode)

# -----------------------
# Checkpoint I/O
# -----------------------
def save_checkpoint(state, checkpoint_dir: str, filename: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))

def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler=None, strict: bool = True, map_to: torch.device = device):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=map_to)
    model.load_state_dict(ck['model_state'], strict=strict)
    if optimizer is not None and 'optimizer_state' in ck:
        optimizer.load_state_dict(ck['optimizer_state'])
    if scheduler is not None and ck.get('scheduler_state') is not None:
        scheduler.load_state_dict(ck['scheduler_state'])
    return ck

def current_lr(optim: torch.optim.Optimizer) -> float:
    return optim.param_groups[0]['lr']

# -----------------------
# Evaluation (uses your local measure.py)
# -----------------------
@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module = None,
             threshold: float = 0.5):
    """
    Returns: avg_loss, avg_dice, avg_iou, avg_acc, avg_sens, avg_spec
    Assumes binary segmentation; model outputs logits (N,1,H,W) or (N,H,W).
    """
    model.eval()
    running_loss = 0.0

    dice_sum = 0.0
    iou_sum = 0.0
    acc_sum = 0.0
    sens_sum = 0.0
    spec_sum = 0.0
    n_batches = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits = model(inputs)
        if logits.dim() == labels.dim() + 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # (N,H,W)

        if criterion is not None:
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        # Your metrics (dice/iou on binarized preds; acc/sens/spec threshold internally at 0.5)
        dice_sum += float(measures.dice_coefficient(preds, labels))
        iou_sum  += float(measures.iou_score(preds, labels))
        acc_sum  += float(measures.accuracy(preds, labels))
        sens_sum += float(measures.sensitivity(preds, labels))
        spec_sum += float(measures.specificity(preds, labels))

        n_batches += 1

    n_batches = max(n_batches, 1)
    avg_loss = running_loss / len(loader.dataset) if (criterion is not None and len(loader.dataset)) else 0.0
    return (
        avg_loss,
        dice_sum / n_batches,
        iou_sum  / n_batches,
        acc_sum  / n_batches,
        sens_sum / n_batches,
        spec_sum / n_batches,
    )

# -----------------------
# Visualization helpers
# -----------------------
def _to_3ch(t: torch.Tensor) -> torch.Tensor:
    # Ensure CHW with 3 channels for grid saving
    if t.dim() == 3:
        t = t.unsqueeze(0)  # (1,H,W) -> (1,1,H,W)
    if t.size(1) == 1:
        t = t.repeat(1, 3, 1, 1)
    return t

def save_visuals(inputs: torch.Tensor,
                 labels: torch.Tensor,
                 probs: torch.Tensor,
                 preds: torch.Tensor,
                 out_dir: str,
                 epoch: int,
                 max_items: int = 4) -> None:
    """
    Saves a grid image showing (per row): Inputs | Labels | Probabilities | Predictions
    - inputs: (N,C,H,W) in [0,1] if using ToTensor
    - labels: (N,H,W) or (N,1,H,W) in {0,1}
    - probs:  (N,H,W) or (N,1,H,W) in [0,1]
    - preds:  (N,H,W) or (N,1,H,W) in {0,1}
    """
    os.makedirs(out_dir, exist_ok=True)

    k = min(max_items, inputs.size(0))
    x  = inputs[:k].detach().cpu()
    y  = labels[:k].detach().cpu()
    pr = probs[:k].detach().cpu()
    pd = preds[:k].detach().cpu()

    if y.dim() == 3:
        y = y.unsqueeze(1)
    if pr.dim() == 3:
        pr = pr.unsqueeze(1)
    if pd.dim() == 3:
        pd = pd.unsqueeze(1)

    # Inputs to 3ch; masks to 3ch for side-by-side viewing
    x3  = _to_3ch(x)
    y3  = _to_3ch(y)
    pr3 = _to_3ch(pr)
    pd3 = _to_3ch(pd)

    # Concatenate along batch to form rows: [x_row | y_row | pr_row | pd_row]
    grid_tensor = torch.cat([x3, y3, pr3, pd3], dim=0)

    # Make a grid with k columns (each column is one sample) and 4 rows (x,y,pr,pd)
    grid = vutils.make_grid(grid_tensor, nrow=k, padding=2, normalize=False)
    out_path = os.path.join(out_dir, f"epoch_{epoch:03d}.png")
    vutils.save_image(grid, out_path)

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
    lr: float = 1e-10,
    weight_decay: float = 1e-4,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,        # set None to disable
    grad_accum_steps: int = 1,
    checkpoint_dir: str = "checkpoints",
    run_name: str = "run",
    early_stopping_patience: Optional[int] = 10,
    threshold: float = 0.5,
    vis_every: int = 1,          # save visuals every N epochs
    vis_max_items: int = 4,      # how many samples to show per grid
    vis_dir: Optional[str] = None,  # where to save; defaults to f"{checkpoint_dir}/vis/{run_name}"
):
    """
    Trains the model and returns a history dict.
    Saves 'best' (by val Dice) and 'last' checkpoints to checkpoint_dir.
    """
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_dice = -math.inf
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_acc": [],
        "val_sens": [],
        "val_spec": [],
        "lr": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (inputs, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                if logits.dim() == labels.dim() + 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)

                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    loss = criterion(logits, labels)
                else:
                    loss = criterion(logits, labels)

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size * grad_accum_steps
            num_samples += batch_size

        scheduler.step()

        # Validation
        val_loss, val_dice, val_iou, val_acc, val_sens, val_spec = evaluate(
            model, val_loader, device, criterion=criterion, threshold=threshold
        )

        # Visualizations (first batch of validation set)
        if vis_dir is None:
            vis_out_dir = os.path.join(checkpoint_dir, "vis", run_name)
        else:
            vis_out_dir = vis_dir

        if (epoch % max(vis_every, 1)) == 0:
            model.eval()
            with torch.inference_mode():
                try:
                    vis_inputs, vis_labels = next(iter(val_loader))
                except StopIteration:
                    vis_inputs, vis_labels = None, None
                if vis_inputs is not None:
                    vis_inputs = vis_inputs.to(device, non_blocking=True)
                    vis_labels = vis_labels.to(device, non_blocking=True).float()
                    logits_vis = model(vis_inputs)
                    if logits_vis.dim() == vis_labels.dim() + 1 and logits_vis.size(1) == 1:
                        logits_vis = logits_vis.squeeze(1)
                    probs_vis = torch.sigmoid(logits_vis)
                    preds_vis = (probs_vis >= threshold).float()
                    # move to cpu inside save_visuals
                    save_visuals(vis_inputs, vis_labels, probs_vis, preds_vis, vis_out_dir, epoch, max_items=vis_max_items)

        # Logging
        train_loss_epoch = running_loss / max(num_samples, 1)
        lr_now = current_lr(optimizer)

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_acc"].append(val_acc)
        history["val_sens"].append(val_sens)
        history["val_spec"].append(val_spec)
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.3e} | "
            f"train_loss={train_loss_epoch:.4f} | val_loss={val_loss:.4f} | "
            f"Dice={val_dice:.4f} | IoU={val_iou:.4f} | "
            f"Acc={val_acc:.4f} | Sens={val_sens:.4f} | Spec={val_spec:.4f}"
        )

        # Checkpoint: last
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "history": history,
                "config": {
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "grad_clip": grad_clip,
                    "grad_accum_steps": grad_accum_steps,
                    "threshold": threshold,
                    "run_name": run_name,
                },
            },
            checkpoint_dir,
            f"{run_name}_last.pt",
        )

        # Checkpoint: best (by Dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "history": history,
                    "config": {
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "grad_clip": grad_clip,
                        "grad_accum_steps": grad_accum_steps,
                        "threshold": threshold,
                        "run_name": run_name,
                    },
                },
                checkpoint_dir,
                f"{run_name}_best.pt",
            )
            print(f"  ↳ New best Dice: {best_val_dice:.4f} (checkpoint saved)")
        else:
            epochs_no_improve += 1

        # Early stopping
        if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered (no improvement in {early_stopping_patience} epochs).")
            break

    return history


history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    criterion=criterion,
    epochs=50,
    lr=1e-4,
    weight_decay=1e-4,
    checkpoint_dir="checkpoints",
    run_name=f"{model_mode}_{dataset_mode}_{loss_mode}",
    early_stopping_patience=10,
    vis_every=1,
    vis_max_items=4,
    vis_dir=None,  # defaults to checkpoints/vis/<run_name>
)