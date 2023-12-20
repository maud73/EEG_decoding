import torch
import torch.nn as nn
import torch.nn.functional as F

def he_init(layer):
  '''
    Variance-preserving initialization of weights for one layer.

    Args:
        layer (torch.nn.Module): layer to initialize
  '''
  if isinstance(layer, nn.Conv2d):
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Code edited from https://github.com/amirhosseinh77/UNet-AerialSegmentation/blob/main/model.py

class DoubleConv(nn.Module):
    """Consists of Convolution -> BatchNorm -> ReLU -> Convolution -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='valid'):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
      return self.double_conv(x)

class Down(nn.Module):
    """Consists of MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Consists of Transpose Convolution -> DoubleConv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x1, x2):
        """x1, x2 (Tensor) : n_batch, channels, height, width"""

        # Crop x2 tensor to match the height and width of x1
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Consists of Convolution -> BatchNorm"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    """Consists of DoubleConv -> (Down)*4 -> (Up)*4 -> Convolution
    Analogous to encoder -> decoder.
    """
    def __init__(self, n_channels, n_classes): # n_classes should be 2, because of binary classification : foreground and background class (gray or black)
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64, kernel_size=3)
        self.down1 = Down(64, 128, kernel_size=3)
        self.down2 = Down(128, 256, kernel_size=3)
        self.down3 = Down(256, 512, kernel_size=2)
        self.down4 = Down(512, 1024, kernel_size=2) # Kernel size of 2 to fit in (height, width)

        self.up1 = Up(1024, 512, kernel_size=2)
        self.up2 = Up(512, 256, kernel_size=2)
        self.up3 = Up(256, 128, kernel_size=3)
        self.up4 = Up(128, 64, kernel_size=(12,57), stride=(4,2))
        self.outc = OutConv(64, n_classes)

        self.apply(he_init)

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
        logits = self.outc(x) # 2 channels containing the probabilities of gray and black
        return logits
    
class UNet_dropout(nn.Module):
    def __init__(self, n_channels, n_classes, p_dropout): # n_classes should be 2, because of binary classification : foreground and background class (gray or black)
        super(UNet_dropout, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.p_dropout = p_dropout

        self.inc = DoubleConv(n_channels, 64, kernel_size=3)
        self.down1 = Down(64, 128, kernel_size=3)
        self.down2 = Down(128, 256, kernel_size=3)
        self.down3 = Down(256, 512, kernel_size=2)
        self.down4 = Down(512, 1024, kernel_size=2) # Kernel size of 2 to fit in (height, width)

        self.up1 = Up(1024, 512, kernel_size=2)
        self.up2 = Up(512, 256, kernel_size=2)
        self.up3 = Up(256, 128, kernel_size=3)
        self.up4 = Up(128, 64, kernel_size=(12,57), stride=(4,2))
        self.outc = OutConv(64, n_classes)
        self.dropout = nn.Dropout2d(p=p_dropout)

        self.apply(he_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(self.dropout(x1))
        x3 = self.down2(self.dropout(x2))
        x4 = self.down3(self.dropout(x3))
        x5 = self.down4(self.dropout(x4))
        x = self.up1(self.dropout(x5), x4)
        x = self.up2(self.dropout(x), x3)
        x = self.up3(self.dropout(x), x2)
        x = self.up4(self.dropout(x), x1)
        logits = self.outc(x) # 2 channels containing the probabilities of gray and black
        return logits
