# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import numpy as np
import torch
import torch.nn as nn
from unet_parts import DoubleConv, Down, Up, OutConv

# x = torch.tensor(np.random.rand(1,3,501,501).astype(np.float32))
# self = UNet(n_channels=3,n_classes=1)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        bl = 8
        self.inc = DoubleConv(n_channels, bl*2**0)
        self.down1 = Down(bl*2**0, bl*2**1)
        self.down2 = Down(bl*2**1, bl*2**2)
        self.down3 = Down(bl*2**2, bl*2**3)
        factor = 2 if bilinear else 1
        self.down4 = Down(bl*2**3, bl*2**4 // factor)
        self.up1 = Up(bl*2**4, bl*2**3 // factor, bilinear)
        self.up2 = Up(bl*2**3, bl*2**2 // factor, bilinear)
        self.up3 = Up(bl*2**2, bl*2**1 // factor, bilinear)
        self.up4 = Up(bl*2**1, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits