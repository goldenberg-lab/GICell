# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

# Function that will the baseline parameter size for UNet model
# path=fn_eosin_new
def find_bl_UNet(path, device, batchnorm=True, start=2, stop=32, step=2):
    preload = torch.load(path)
    should_stop = False
    for bl in range(start, stop+1, step):
        mdl = UNet(3, 1, bl, batchnorm)
        mdl.to(device)
        try:
            mdl.load_state_dict(preload)
            should_stop = True
            print('Model successfully loaded for bl=%i' % bl)
        except:
            print('Model will not load parameters for bl=%i' % bl)
        if should_stop:
            break
    return mdl


# x = torch.tensor(np.random.rand(1,3,501,501)).double()
# self = UNet(n_channels=3,n_classes=1)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bl = 8, batchnorm=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, bl*2**0, batchnorm)
        self.down1 = Down(bl*2**0, bl*2**1, batchnorm)
        self.down2 = Down(bl*2**1, bl*2**2, batchnorm)
        self.down3 = Down(bl*2**2, bl*2**3, batchnorm)
        self.down4 = Down(bl*2**3, bl*2**4, batchnorm)
        self.up1 = Up(bl*2**4, bl*2**3, batchnorm)
        self.up2 = Up(bl*2**3, bl*2**2, batchnorm)
        self.up3 = Up(bl*2**2, bl*2**1, batchnorm)
        self.up4 = Up(bl*2**1, 64, batchnorm)
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, batchnorm):
        super().__init__()
        mid_channels = out_channels
        if batchnorm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels,track_running_stats=True, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels,track_running_stats=True, momentum=0.1),
                nn.ReLU(inplace=True))
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batchnorm):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, batchnorm):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, batchnorm)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
