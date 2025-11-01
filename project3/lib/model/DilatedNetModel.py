import torch.nn.functional as F
import torch.nn as nn
import torch


class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder with dilations instead of pooling
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1, dilation=1)
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)

        # Bottleneck
        self.bottleneck = nn.Conv2d(64, 64, 3, padding=16, dilation=16)

        # Decoder (upsampling with transpose convs)
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)  # output mask

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))
        e3 = F.relu(self.enc_conv3(e2))

        # Bottleneck
        b = F.relu(self.bottleneck(e3))

        # Decoder with skip connections
        d0 = F.relu(self.dec_conv0(torch.cat((e3, self.up0(b)), dim=1)))
        d1 = F.relu(self.dec_conv1(torch.cat((e2, self.up1(d0)), dim=1)))
        d2 = F.relu(self.dec_conv2(torch.cat((e1, self.up2(d1)), dim=1)))
        d3 = self.dec_conv3(torch.cat((e0, self.up3(d2)), dim=1))

        return d3
