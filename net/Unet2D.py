from .Unet2DBlock import *
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Unet2D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Unet2D, self).__init__()
        self.n_classes = n_classes
        self.inc = ConvDouble(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, self.n_classes, kernel_size=1)

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
        x = self.outc(x)
        if self.n_classes == 1:
            x = F.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)

        return x