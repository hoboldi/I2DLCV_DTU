import argparse
from pathlib import Path
import os
import json
from datetime import datetime
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

# -----------------------
# Config (quick flags)
# -----------------------
model_mode   = "unet"
dataset_mode = "drive"
loss_mode    = "bce"

data_root = Path("data/DRIVE")
tfm_train = T.Compose([T.ToTensor()])
tfm_val   = T.Compose([T.ToTensor()])

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
def import_class(primary: str, fallback: str, attr: str):
    """
    Import attr from primary module path, else from fallback.
    E.g., import_class("Project3.lib.model", "lib.model", "UNetModel")
    """
    try:
        mod = __import__(primary, fromlist=[attr])
    except Exception:
        mod = __import__(fallback, fromlist=[attr])
    return getattr(mod, attr)

def import_module(primary: str, fallback: str):
    try:
        return __import__(primary, fromlist=["*"])
    except Exception:
        return __import__(fallback, fromlist=["*"])

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

train_ds = DatasetClass(root_dir=dataset_root_dir, train=True,  transform=tfm_train)
val_ds   = DatasetClass(root_dir=dataset_root_dir, train=False, transform=tfm_val)

# Example loaders (tweak batch sizes/workers as needed)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# -----------------------
# Losses
# -----------------------
losses = import_module("Project3.lib.losses", "lib.losses")

def make_criterion(mode: str) -> nn.Module:
    mode = mode.lower()
    if mode == "bce":
        # Prefer BCEWithLogitsLoss (numerically stable) for binary seg
        return nn.BCEWithLogitsLoss()
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
# Metrics for segmentation
# -----------------------
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    # pred, target expected as binary {0,1} float tensors
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (2.0 * inter + eps) / (denom + eps)

def iou_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return (inter + eps) / (union + eps)

# -----------------------
# Evaluation
# -----------------------
@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module = None,
             threshold: float = 0.5):
    """
    Returns (avg_loss, avg_dice, avg_iou).
    - If criterion is None, only metrics are computed.
    - Assumes binary segmentation: model outputs logits with shape (N,1,H,W) or (N,H,W).
    """
    model.eval()
    running_loss = 0.0
    n_pixels = 0
    dice_sum, iou_sum = 0.0, 0.0
    n_batches = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)

        # Expect labels as {0,1}; cast to float for BCE/Dice
        labels = labels.to(device, non_blocking=True).float()

        logits = model(inputs)
        if logits.dim() == labels.dim() + 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # (N,H,W)

        batch_loss = 0.0
        if criterion is not None:
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                batch_loss = criterion(logits, labels)
            else:
                # If custom losses expect probabilities, pass sigmoid(logits)
                probs = torch.sigmoid(logits)
                batch_loss = criterion(probs, labels)
            running_loss += batch_loss.item() * inputs.size(0)

        # Threshold for metrics
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        dice_sum += dice_coeff(preds, labels)
        iou_sum  += iou_coeff(preds, labels)
        n_batches += 1
        n_pixels  += inputs.size(0) * inputs.shape[-2] * inputs.shape[-1]

    avg_loss = running_loss / len(loader.dataset) if (criterion is not None and len(loader.dataset)) else 0.0
    avg_dice = dice_sum / max(n_batches, 1)
    avg_iou  = iou_sum  / max(n_batches, 1)
    return avg_loss, avg_dice, avg_iou

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epochs: int = 50,
    lr: float = 1e-3,
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
):
    """
    Trains the model and returns a history dict.
    Saves 'best' (by val Dice) and 'last' checkpoints to checkpoint_dir.
    """
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # If no scheduler provided, use a simple cosine annealing (nice default)
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
                # Match training loss behavior to your evaluate() logic
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    loss = criterion(logits, labels)
                else:
                    probs = torch.sigmoid(logits)
                    loss = criterion(probs, labels)

            # Gradient accumulation (optional)
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
            running_loss += loss.item() * batch_size * grad_accum_steps  # undo division for logging
            num_samples += batch_size

        # Step scheduler once per epoch
        scheduler.step()

        # Validation
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device, criterion=criterion, threshold=threshold)

        # Logging
        train_loss_epoch = running_loss / max(num_samples, 1)
        lr_now = current_lr(optimizer)

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["lr"].append(lr_now)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"lr={lr_now:.3e} | "
            f"train_loss={train_loss_epoch:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f}"
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

'''
history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    criterion=criterion,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    checkpoint_dir="checkpoints",
    run_name=f"{model_mode}_{dataset_mode}_{loss_mode}",
    early_stopping_patience=10,
)
'''