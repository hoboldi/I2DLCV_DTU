import os
import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from Project2.dual_stream import TwoStream
    from Project2.datasets import FrameImageDataset
except Exception:
    # allow running when this file is executed from Project2/ directly
    from dual_stream import TwoStream
    from datasets import FrameImageDataset


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, device, loader, optimizer, criterion, use_flow):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        # unpack batch depending on dataset output
        if use_flow:
            imgs, flows, labels = batch
            imgs = imgs.to(device)
            flows = flows.to(device)
            labels = labels.to(device)
            logits = model(imgs, flows)
        else:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, device, loader, criterion, use_flow):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if use_flow:
                imgs, flows, labels = batch
                imgs = imgs.to(device)
                flows = flows.to(device)
                labels = labels.to(device)
                logits = model(imgs, flows)
            else:
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)

            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def make_dataloaders(root_dir, batch_size, num_workers, use_flow, flow_root=None, img_size=(64, 64)):
    from torchvision import transforms as T

    transform = T.Compose([T.Resize(img_size), T.ToTensor()])

    train_ds = FrameImageDataset(root_dir=root_dir, split='train', transform=transform,
                                 use_flow=use_flow, flow_root=flow_root)
    val_ds = FrameImageDataset(root_dir=root_dir, split='val', transform=transform,
                               use_flow=use_flow, flow_root=flow_root)
    test_ds = FrameImageDataset(root_dir=root_dir, split='test', transform=transform,
                               use_flow=use_flow, flow_root=flow_root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)



    return train_loader, val_loader, test_loader


def save_checkpoint(state, path: str):
    torch.save(state, path)


def main(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print('Using device:', device)

    train_loader, val_loader, test_loader = make_dataloaders(args.root_dir, args.batch_size, args.num_workers,
                                               args.use_flow, flow_root=args.flow_root, img_size=(args.img_size, args.img_size))

    model = TwoStream(num_classes=args.num_classes, dropout_rate=args.dropout, use_batchnorm=args.use_bn, avgLogits=args.avg_logits)
    if args.use_flow:
        assert args.flow_in_channels is not None, 'flow_in_channels must be provided when use_flow is True'
        model.init_flow_stream(args.flow_in_channels, use_batchnorm=args.use_bn)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, args.use_flow)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion, args.use_flow)
        scheduler.step()

        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs}  time={t1-t0:.1f}s  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'args': vars(args)
        }
        save_checkpoint(ckpt, str(output_dir / f'checkpoint_epoch_{epoch}.pt'))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ckpt, str(output_dir / 'best_checkpoint.pt'))

    print('Training finished. Best val acc:', best_val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/dtu/datasets1/02516/ucf101_noleakage')
    parser.add_argument('--flow_root', type=str, default=None)
    parser.add_argument('--use_flow', action='store_true')
    parser.add_argument('--flow_in_channels', type=int, default=None,
                        help='number of channels in flow input (e.g. 2*9=18)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--avg_logits', dest='avg_logits', action='store_true', help='Use averaging of logits (default)')
    parser.add_argument('--no-avg_logits', dest='avg_logits', action='store_false', help='Use SVM head instead of averaging')
    parser.set_defaults(avg_logits=True)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    main(args)
