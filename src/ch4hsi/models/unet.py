"""Compact U-Net for plume segmentation (any number of input channels)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout, dropout=0.0):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.stem = DoubleConv(in_ch, ch[0])
        self.down = nn.ModuleList([DoubleConv(ch[i], ch[i + 1], dropout if i == depth - 1 else 0.0) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.depth = depth

    def forward(self, x):
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = up(h)
            s = skips.pop()
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = dec(torch.cat([h, s], dim=1))
        return self.head(h)


def build_model(tcfg, in_ch: int) -> nn.Module:
    if tcfg.model == "unet":
        return UNet(in_ch, tcfg.base_channels, tcfg.depth, tcfg.dropout)
    if tcfg.model == "smp":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=tcfg.smp_encoder, encoder_weights=None, in_channels=in_ch, classes=1)
    raise ValueError(f"unknown model {tcfg.model}")
