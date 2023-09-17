import torch
import torch.nn as nn

class ConvDouble(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDouble, self).__init__()
        self.conv_double = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_double(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvDouble(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down_block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv_double = ConvDouble(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.conv_trans(x1)
        x = torch.cat((x2, x1), dim=1)
        return self.conv_double(x)