import torch
import torch.nn as nn
import torch.nn.functional as F


# Optimized CNN Architecture for Regularization Experiments (~150k parameters)
class Network(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0, use_batchnorm=False, in_channels =3, img_size=64):
        super(Network, self).__init__()

        # More channels with Global Average Pooling: 3->32->64->128
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # 128x128 -> 128x128
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64x64 -> 64x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization layers (optional)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.bn4 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)  # Halves dimensions each time


        # But we'll do 4 pools to get to reasonable size: 128->64->32->16->8
        self.flatten = nn.Flatten()
        self.fc1S = nn.Linear(128 * img_size/(2**4), 64) # 128 channels, image size halves each maxpool
        self.fc2S = nn.Linear(64, 16)
        self.fc3S = nn.Linear(16, num_classes)

        # Configurable dropout
        self.dropout_rateS = dropout_rate
        self.dropoutS = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        # ----- Spatial stream -----
        s = self.poolS(F.relu(self.bn1S(self.conv1(x))))
        s = self.poolS(F.relu(self.bn2S(self.conv1(s))))
        s = self.poolS(F.relu(self.bn3S(self.conv3(s))))
        s = self.poolS(F.relu(self.bn4S(self.conv4(s))))  # 128→64→32→16→8

        s = self.flattenS(s)  # (N, 128*8*8)
        s = self.dropoutS(F.relu(self.fc1(s)))
        s = self.dropoutS(F.relu(self.fc2(s)))
        logits_s = self.fc3S(s)  # (N, num_classes)


        return logits_s

    