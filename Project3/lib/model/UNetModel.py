# lib/model/UNetModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNetModel"]


# ---------- blocks ----------
class DoubleConv(nn.Module):
    """(Conv->Norm->ReLU)x2 with optional residual when in_ch==out_ch"""
    def __init__(self, in_ch, out_ch, norm="bn", residual=True, dropout=0.0):
        super().__init__()
        Norm = {
            "bn": nn.BatchNorm2d,
            "gn": lambda c: nn.GroupNorm(num_groups=min(32, c), num_channels=c),
            "ln": lambda c: nn.GroupNorm(num_groups=1, num_channels=c),  # channel-wise LN-ish
            None: None,
        }[norm]

        layers = []
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=(Norm is None))]
        if Norm is not None: layers += [Norm(out_ch)]
        layers += [nn.ReLU(inplace=True)]
        if dropout > 0: layers += [nn.Dropout2d(dropout)]

        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=(Norm is None))]
        if Norm is not None: layers += [Norm(out_ch)]
        layers += [nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*layers)
        self.residual = residual and (in_ch == out_ch)

    def forward(self, x):
        y = self.block(x)
        return y + x if self.residual else y


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, **dc_kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(in_ch, out_ch, **dc_kwargs)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsample + concat(skip) + DoubleConv; handles odd sizes via padding.
    """
    def __init__(self, in_ch, out_ch, bilinear=True, **dc_kwargs):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch, **dc_kwargs)
        else:
            # learnable upsampling; halve channels during up
            mid = in_ch // 2
            self.up = nn.ConvTranspose2d(in_ch - (in_ch - mid), mid, kernel_size=2, stride=2)
            # NOTE: we will concat with skip (mid + skip_ch) externally; keep conv generic
            self.conv = DoubleConv(in_ch, out_ch, **dc_kwargs)

    @staticmethod
    def _match(ref, x):
        dy = ref.size(2) - x.size(2)
        dx = ref.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return x

    def forward(self, x, skip):
        if self.bilinear:
            x = self.up(x)
        else:
            x = self.up(x)
        x = self._match(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---------- model ----------
class UNetModel(nn.Module):
    """
    U-Net (safe sizes, norm+residual, optional dropout).
    Returns logits (N,1,H,W); works with your FocalLoss/Dice/BCE-with-logits.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,   # keep =1 for your pipeline (single-logit)
        base_ch: int = 64,
        bilinear: bool = True,
        norm: str = "bn",        # "bn" | "gn" | "ln" | None
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pad_multiple = 16  # robust to arbitrary HxW

        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*2, base_ch*2, base_ch*4

        self.inc   = DoubleConv(in_channels, c1, norm=norm, residual=residual, dropout=dropout)
        self.down1 = Down(c1, c2, norm=norm, residual=residual, dropout=dropout)
        self.down2 = Down(c2, c3, norm=norm, residual=residual, dropout=dropout)
        self.down3 = Down(c3, c4, norm=norm, residual=residual, dropout=dropout)
        self.down4 = Down(c4, c5, norm=norm, residual=residual, dropout=dropout)

        # Decoder: Up expects in_ch = skip_ch + up_ch
        self.up1 = Up(in_ch=c4 + c5, out_ch=c4, bilinear=bilinear, norm=norm, residual=residual, dropout=dropout)
        self.up2 = Up(in_ch=c3 + c4, out_ch=c3, bilinear=bilinear, norm=norm, residual=residual, dropout=dropout)
        self.up3 = Up(in_ch=c2 + c3, out_ch=c2, bilinear=bilinear, norm=norm, residual=residual, dropout=dropout)
        self.up4 = Up(in_ch=c1 + c2, out_ch=c1, bilinear=bilinear, norm=norm, residual=residual, dropout=dropout)

        self.outc = OutConv(c1, out_channels)

        self._init_weights()

    # ---- padding helpers ----
    @staticmethod
    def _to_multiple(x, mult):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        return pad_h, pad_w

    def _pad(self, x):
        ph, pw = self._to_multiple(x, self.pad_multiple)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, ph, pw

    def _unpad(self, x, ph, pw):
        if ph or pw:
            x = x[..., : x.shape[-2] - ph, : x.shape[-1] - pw]
        return x

    # ---- init ----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- forward ----
    def forward(self, x):
        # pad -> encode
        x, ph, pw = self._pad(x)
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b  = self.down4(s4)

        # decode (upsample inside Up; it resizes to match skip safely)
        x = self.up1(b,  s4)
        x = self.up2(x,  s3)
        x = self.up3(x,  s2)
        x = self.up4(x,  s1)

        logits = self.outc(x)
        logits = self._unpad(logits, ph, pw)  # back to original HxW
        return logits