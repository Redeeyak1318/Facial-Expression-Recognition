import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResidualCNN(nn.Module):
    """
    V2-B Lightweight Residual CNN architecture.
    """
    def __init__(self, num_classes=7):
        super(ResidualCNN, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Residual Block 1: 32 -> 32, stride 1
        self.res1 = ResidualBlock(32, 32, stride=1)
        
        # MaxPool 2x2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual Block 2: 32 -> 64, stride 2 (projection shortcut)
        self.res2 = ResidualBlock(32, 64, stride=2)
        
        # Residual Block 3: 64 -> 128, stride 2 (projection shortcut)
        self.res3 = ResidualBlock(64, 128, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.30)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        
        out = self.res1(out)
        out = self.pool1(out)
        
        out = self.res2(out)
        out = self.res3(out)
        
        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)
        
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
