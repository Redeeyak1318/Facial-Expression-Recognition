import torch
import torch.nn as nn

class EnhancedCNN(nn.Module):
    """
    V2-A Enhanced CNN architecture with BatchNorm and a deeper 4th block.
    
    Architecture:
    Input: 1x48x48
    - Block 1: Conv(1->32, k=3, pad=1) -> BatchNorm2d(32) -> ReLU -> MaxPool(2x2)
    - Block 2: Conv(32->64, k=3, pad=1) -> BatchNorm2d(64) -> ReLU -> MaxPool(2x2)
    - Block 3: Conv(64->128, k=3, pad=1) -> BatchNorm2d(128) -> ReLU -> MaxPool(2x2)
    - Block 4: Conv(128->256, k=3, pad=1) -> BatchNorm2d(256) -> ReLU
    - Global Average Pooling -> 256
    - Dropout(p=0.30)
    - Linear(256 -> 7)
    """
    def __init__(self, num_classes: int = 7):
        super(EnhancedCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 48, 48)
            
        Returns:
            torch.Tensor: Logits of shape (B, 7)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_avg_pool(x)
        
        # Flatten the tensor from (B, 256, 1, 1) to (B, 256)
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)
        
        return x
