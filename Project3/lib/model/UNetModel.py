import torch.nn.functional as F
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_conv00 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_conv01 = nn.Conv2d(64,64, 3, padding=1)

        self.pool0 = nn.MaxPool2d(2,2) # 128 -> 64

        self.enc_conv10 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv11 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2,2) # 64 -> 32

        self.enc_conv20 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_conv21 = nn.Conv2d(128,128, 3, padding=1)

        self.pool2 = nn.MaxPool2d(2,2) # 32 -> 16

        self.enc_conv30 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc_conv31 = nn.Conv2d(128,128, 3, padding=1)

        self.pool3 = nn.MaxPool2d(2,2)  # 16 -> 8

        # Bottleneck
        self.bottleneck_conv0 = nn.Conv2d(128,256, 3, padding=1)
        self.bottleneck_conv1 = nn.Conv2d(256, 256, 3, padding=1)

        # Decoder
        self.upsample0 = nn.ConvTranspose2d(256, 128, kernel_size=2, padding=0, stride=2) # 8 -> 16

        self.dec_conv00 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec_conv01 = nn.Conv2d(128,128, 3, padding=1)

        self.upsample1 = nn.ConvTranspose2d(128, 128, 2, padding=0, stride=2) # 16 -> 32

        self.dec_conv10 = nn.Conv2d(256,128, 3, padding=1)
        self.dec_conv11 = nn.Conv2d(128,128, 3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(128,128, 2, padding=0, stride=2)

        self.dec_conv20 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec_conv21 = nn.Conv2d(128, 64, 3, padding=1)

        self.upsample3 = nn.ConvTranspose2d(64, 64, 2, padding=0, stride=2)

        self.dec_conv30 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_conv31 = nn.Conv2d(64, 64, 3, padding=1)
        self.dec_conv32 = nn.Conv2d(64, 1, 1, padding=0)

    def forward(self, x):

        # Encoder
        e00 = F.relu(self.enc_conv00(x))
        e01 = F.relu(self.enc_conv01(e00))

        p0 = self.pool0(e01)

        e10 = F.relu(self.enc_conv10(p0))
        e11 = F.relu(self.enc_conv11(e10))

        p1 = self.pool1(e11)

        e20 = F.relu(self.enc_conv20(p1))
        e21 = F.relu(self.enc_conv21(e20))

        p2 = self.pool2(e21)

        e30 = F.relu(self.enc_conv30(p2))
        e31 = F.relu(self.enc_conv30(e30))

        p3 = self.pool3(e31)

        b0 = F.relu(self.bottleneck_conv0(p3))
        b1 = F.relu(self.bottleneck_conv1(b0))

        u0 = self.upsample0(b1)

        d00 = F.relu(self.dec_conv00(torch.cat((u0,e31), dim=1)))
        d01 = F.relu(self.dec_conv01(d00))

        u1 = self.upsample1(d01)

        d10 = F.relu(self.dec_conv10(torch.cat((u1,e21),dim=1)))
        d11 = F.relu(self.dec_conv11(d10))

        u2 = self.upsample2(d11)

        d20 = F.relu(self.dec_conv20(torch.cat((u2,e11),dim=1)))
        d21 = F.relu(self.dec_conv21(d20))

        u3 = self.upsample3(d21)

        d30 = F.relu(self.dec_conv30(torch.cat((u3,e01),dim=1)))
        d31 = F.relu(self.dec_conv31(d30))
        d32 = F.relu(self.dec_conv32(d31))

        return d32


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
        self.dec_conv3 = nn.Conv2d(128, 2, 1, padding=0)  # final output

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