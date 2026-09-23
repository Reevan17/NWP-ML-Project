import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """Channel Attention Mechanism: Dynamically weights feature channels"""
    def __init__(self, channels, reduction=8):
        super(SqueezeExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights

class DoubleConv(nn.Module):
    """Double 3x3 Conv with BatchNorm, LeakyReLU, and SE Attention"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.se = SqueezeExcitation(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x)) + self.shortcut(x)

class WeatherUNet(nn.Module):
    """
    Residual Attention Weather U-Net (ResAttUNet)
    Learns the fine-grained atmospheric residual correction field:
    T_corrected = T_GFS + Delta_T
    """
    def __init__(self, in_channels=5, out_channels=1):
        super(WeatherUNet, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        # Output delta layer
        self.outc = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Input channel 0 is GFS Temperature
        gfs_temp = x[:, 0:1, :, :]
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        b = self.bottleneck(x4)
        
        d3 = self.up1(b)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv1(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv2(d2)
        
        d1 = self.up3(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv3(d1)
        
        delta_t = self.outc(d1)
        
        # Residual Addition: Output = Input Temperature + Learned Residual Bias
        return gfs_temp + delta_t