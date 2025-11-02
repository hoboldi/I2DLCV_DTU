import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import os
import json
from datetime import datetime

try:
    from Project2.dual_stream import Network
    from Project2.datasets import FrameImageDataset
except Exception:
    # allow running when this file is executed from Project2/ directly
    from dual_stream import Network
    from datasets import FrameImageDataset


def save_checkpoint(state, checkpoint_dir: str, filename: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, scheduler=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model_state'])
    if optimizer is not None and 'optimizer_state' in ck:
        optimizer.load_state_dict(ck['optimizer_state'])
    if scheduler is not None and 'scheduler_state' in ck and ck['scheduler_state'] is not None:
        scheduler.load_state_dict(ck['scheduler_state'])
    return ck

# helper to read current LR
def current_lr(optim: torch.optim.Optimizer) -> float:
    return optim.param_groups[0]['lr']


def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Run evaluation on loader and return (loss, accuracy_percent)."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels).item()
            n_samples += inputs.size(0)

    avg_loss = running_loss / n_samples if n_samples else 0.0
    acc = 100.0 * running_corrects / n_samples if n_samples else 0.0
    return avg_loss, acc


def train(args):
    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')

    # transforms for spatial frames
    train_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])
    val_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])

    # dataset and dataloader (frame image dataset)
    train_dataset = FrameImageDataset(root_dir=args.root_dir, split=args.split, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # validation dataloader (optional)
    val_loader = None
    if args.evaluate:
        val_dataset = FrameImageDataset(root_dir=args.root_dir, split=args.eval_split, transform=val_transform)
        eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
        val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, num_workers=max(1, args.workers//2), pin_memory=True)

    # model
    model = Network(num_classes=args.num_classes, dropout_rate=args.dropout, img_size=args.img_size, in_channels=args.in_channels)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        threshold=args.plateau_threshold,
        min_lr=args.plateau_min_lr,
        verbose=True,
    )
    # ================================================

    start_epoch = 0
    # optionally resume
    if args.resume:
        ck = load_checkpoint(args.resume, model, optimizer, scheduler)  # pass scheduler
        start_epoch = ck.get('epoch', 0) + 1
        print(f"Resumed from checkpoint {args.resume} at epoch {start_epoch} (lr: {current_lr(optimizer):.6f})")

    # metadata logging
    run_dir = Path(args.checkpoint_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # track best validation accuracy (fallback to training accuracy if no val set)
    best_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        n_samples = 0
        epoch_start = time.time()

        lr_before = current_lr(optimizer)  # snapshot LR at epoch start

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels).item()
            n_samples += inputs.size(0)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = running_loss / n_samples if n_samples else 0.0
                acc = 100.0 * running_corrects / n_samples if n_samples else 0.0
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"loss: {avg_loss:.4f} acc: {acc:.2f}% lr: {lr_before:.6f}")

        epoch_loss = running_loss / n_samples if n_samples else 0.0
        epoch_acc = 100.0 * running_corrects / n_samples if n_samples else 0.0
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s - loss: {epoch_loss:.4f} acc: {epoch_acc:.2f}% lr(before): {lr_before:.6f}")

        # === Evaluate BEFORE scheduler.step so plateau monitors correct metric ===
        val_loss = None
        val_acc = None
        if args.evaluate and val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Validation - epoch: {epoch} loss: {val_loss:.4f} acc: {val_acc:.2f}% (lr(before): {lr_before:.6f})")

        metric_for_lr = val_loss if (val_loss is not None) else epoch_loss
        scheduler.step(metric_for_lr)  # reduce on plateau
        lr_after = current_lr(optimizer)
        # ================================================================

        # checkpoint (save epoch)
        ck_name = f"checkpoint_epoch_{epoch}.pth"
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr_after,  # store post-step LR
        }, str(run_dir), ck_name)

        # run evaluation-based best saving (or fallback to train acc)
        if args.evaluate and val_loader is not None:
            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'train_loss': epoch_loss,
                    'train_acc': epoch_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': lr_after,
                }, str(run_dir), 'best_spatial.pth')
        else:
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                save_checkpoint({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'train_loss': epoch_loss,
                    'train_acc': epoch_acc,
                    'lr': lr_after,
                }, str(run_dir), 'best_spatial.pth')

        if lr_after != lr_before:
            print(f"LR reduced: {lr_before:.6f} -> {lr_after:.6f}")

    print(f"Training completed. checkpoints saved to {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train spatial model on FrameImageDataset')
    parser.add_argument('--root-dir', type=str, default='/dtu/datasets1/02516/ucf101_noleakage')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--in-channels', type=int, default=3,
                        help='number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--dropout', type=float, default=0.3)

    # kept for backwards-compat (unused by ReduceLROnPlateau)
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--log-interval', type=int, default=20)
    # evaluation options
    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.set_defaults(evaluate=True)
    parser.add_argument('--eval-split', type=str, default='val')
    parser.add_argument('--eval-batch-size', type=int, default=None)

    # NEW: plateau scheduler args
    parser.add_argument('--plateau-patience', type=int, default=5,
                        help='epochs with no improvement before reducing LR')
    parser.add_argument('--plateau-factor', type=float, default=0.5,
                        help='LR reduction factor (new_lr = lr * factor)')
    parser.add_argument('--plateau-threshold', type=float, default=1e-3,
                        help='minimum significant improvement in monitored metric')
    parser.add_argument('--plateau-min-lr', type=float, default=1e-6,
                        help='lower bound on the learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)