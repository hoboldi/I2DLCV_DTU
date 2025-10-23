import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F



# Optimized CNN Architecture for Regularization Experiments (~150k parameters)
class TwoStream(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0, use_batchnorm=False):
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

    def forward(self, xS):
        # ----- Spatial stream -----
        s = self.poolS(F.relu(self.bn1S(self.conv1S(xS))))
        s = self.poolS(F.relu(self.bn2S(self.conv2S(s))))
        s = self.poolS(F.relu(self.bn3S(self.conv3S(s))))
        s = self.poolS(F.relu(self.bn4S(self.conv4S(s))))  # 128→64→32→16→8

        s = self.flattenS(s)  # (N, 128*8*8)
        s = self.dropoutS(F.relu(self.fc1S(s)))
        s = self.dropoutS(F.relu(self.fc2S(s)))
        logits_s = self.fc3S(s)  # (N, num_classes)

        logits = logits_s 

        return logits