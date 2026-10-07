import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.cbam = CBAM(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.cbam(out)
        return out + residual

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, 1, bias=False), nn.BatchNorm2d(out_channels // 4), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(out_channels // 4), nn.GELU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(out_channels // 4), nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 4, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels // 4), nn.GELU())
        self.out_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d3 = self.conv3(x)
        d4 = self.conv4(x)
        return self.out_conv(torch.cat([d1, d2, d3, d4], dim=1))

class WeatherUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1):
        super(WeatherUNet, self).__init__()
        self.inc = DoubleConv(in_channels, 48)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(48, 96))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(96, 192))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(192, 384))
        
        self.aspp = ASPP(384, 384)
        
        self.up1 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(384, 192)
        
        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(192, 96)
        
        self.up3 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(96, 48)
        
        self.outc = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        b = self.aspp(x4)
        
        d3 = self.up1(b)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.conv1(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.conv2(d2)
        
        d1 = self.up3(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.conv3(d1)
        
        return self.outc(d1)