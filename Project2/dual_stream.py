import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class No_Leakage(torch.utils.data.Dataset):
    def __init__(self, split, transform, data_path='/dtu/datasets1/02516/ucf101_noleakage'):
        'Initialization'
        self.transform = transform
        os.path.join(data_path, split)

        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path + '/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


def preprocess(size=128,batch_size=64):
    train_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor()])

    trainset = No_Leakage(split='train', transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)

    valset = No_Leakage(split='val', transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=3)

    testset = No_Leakage(split='test', transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)


# Optimized CNN Architecture for Regularization Experiments (~150k parameters)
class TwoStream(nn.Module):
    def __init__(self, xS, xT, num_classes=10, dropout_rate=0.0, use_batchnorm=False):
        super(TwoStream, self).__init__()

        # More channels with Global Average Pooling: 3->32->64->128
        self.conv1S = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 128x128 -> 128x128
        self.conv2S = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64 -> 64x64
        self.conv3S = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv4S = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization layers (optional)
        self.bn1S = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.bn2S = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.bn3S = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.bn4S = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()

        self.poolS = nn.MaxPool2d(2, 2)  # Halves dimensions each time

        # With pooling after each conv: 128->64->32->16->8->4->2 after 6 pools
        # But we'll do 4 pools to get to reasonable size: 128->64->32->16->8
        self.flattenS = nn.Flatten()

        # Compact FC head for ~150k total parameters: 8*8*128 -> 64 -> 16 -> 2
        self.fc1S = nn.Linear(128 * 8 * 8, 64)
        self.fc2S = nn.Linear(64, 16)
        self.fc3S = nn.Linear(16, num_classes)

        # Configurable dropout
        self.dropout_rateS = dropout_rate
        self.dropoutS = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()


        # More channels with Global Average Pooling: 3->32->64->128
        self.conv1T = nn.Conv2d(18, 32, kernel_size=3, padding=1)  # 128x128 -> 128x128
        self.conv2T = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64 -> 64x64
        self.conv3T = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv4T = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization layers (optional)
        self.bn1T = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.bn2T = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.bn3T = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.bn4T = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()

        self.poolT = nn.MaxPool2d(2, 2)  # Halves dimensions each time

        # With pooling after each conv: 128->64->32->16->8->4->2 after 6 pools
        # But we'll do 4 pools to get to reasonable size: 128->64->32->16->8
        self.flattenT = nn.Flatten()

        # Compact FC head for ~150k total parameters: 8*8*128 -> 64 -> 16 -> 2
        self.fc1T = nn.Linear(128 * 8 * 8, 64)
        self.fc2T = nn.Linear(64, 16)
        self.fc3T = nn.Linear(16, num_classes)

        # Configurable dropout
        self.dropout_rateT = dropout_rate
        self.dropoutT = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, xS, xT):
        # ----- Spatial stream -----
        s = self.poolS(F.relu(self.bn1S(self.conv1S(xS))))
        s = self.poolS(F.relu(self.bn2S(self.conv2S(s))))
        s = self.poolS(F.relu(self.bn3S(self.conv3S(s))))
        s = self.poolS(F.relu(self.bn4S(self.conv4S(s))))  # 128→64→32→16→8

        s = self.flattenS(s)  # (N, 128*8*8)
        s = self.dropoutS(F.relu(self.fc1S(s)))
        s = self.dropoutS(F.relu(self.fc2S(s)))
        logits_s = self.fc3S(s)  # (N, num_classes)

        # ----- Temporal stream -----
        t = self.poolT(F.relu(self.bn1T(self.conv1T(xT))))
        t = self.poolT(F.relu(self.bn2T(self.conv2T(t))))
        t = self.poolT(F.relu(self.bn3T(self.conv3T(t))))
        t = self.poolT(F.relu(self.bn4T(self.conv4T(t))))  # 128→64→32→16→8

        t = self.flattenT(t)  # (N, 128*8*8)
        t = self.dropoutT(F.relu(self.fc1T(t)))
        t = self.dropoutT(F.relu(self.fc2T(t)))
        logits_t = self.fc3T(t)

        logits = 0.5 * (logits_s + logits_t)  # late fusion (average)

        return logits


def train_epoch(model, train_loader, criterion, optimizer, device, epoch_num):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create tqdm progress bar for training
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch_num} [Train]', leave=False)

    for batch_idx, (data, target) in enumerate(train_pbar):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        current_acc = 100. * correct / total
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device, epoch_num):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # Create tqdm progress bar for validation
    val_pbar = tqdm(test_loader, desc=f'Epoch {epoch_num} [Val]', leave=False)

    with torch.no_grad():
        for data, target in val_pbar:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()

            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            current_acc = 100. * correct / total
            val_pbar.set_postfix({
                'Loss': f'{test_loss / (val_pbar.n + 1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc