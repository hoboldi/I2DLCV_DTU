import torch.nn.functional as F
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(torch.cat((e3,self.upsample0(b)),dim=1)))
        d1 = F.relu(self.dec_conv1(torch.cat((e2,self.upsample1(d0)),dim=1)))
        d2 = F.relu(self.dec_conv2(torch.cat((e1,self.upsample2(d1)),dim=1)))
        d3 = self.dec_conv3(torch.cat((e0,self.upsample3(d2)),dim=1))  # no activation
        return d3


class UNet2(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: Conv + Strided Conv instead of pooling
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.down0 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # replaces pool0

        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # replaces pool1

        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # replaces pool2

        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.down3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # replaces pool3

        # Bottleneck
        self.bottleneck = nn.Conv2d(64, 64, 3, padding=1)

        # Decoder: Transposed convs instead of upsampling
        self.up0 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # replaces upsample0
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)  # final output

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(self.down0(e0)))
        e2 = F.relu(self.enc_conv2(self.down1(e1)))
        e3 = F.relu(self.enc_conv3(self.down2(e2)))
        b = F.relu(self.bottleneck(self.down3(e3)))

        # Decoder
        d0 = F.relu(self.dec_conv0(torch.cat((e3, self.up0(b)), dim=1)))
        d1 = F.relu(self.dec_conv1(torch.cat((e2, self.up1(d0)), dim=1)))
        d2 = F.relu(self.dec_conv2(torch.cat((e1, self.up2(d1)), dim=1)))
        d3 = self.dec_conv3(torch.cat((e0, self.up3(d2)), dim=1))
        return d3