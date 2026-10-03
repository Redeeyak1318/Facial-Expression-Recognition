"""
Module defining the Lightweight CNN architecture for Facial Expression Recognition.
"""
import torch
import torch.nn as nn

class LightweightCNN(nn.Module):
    """
    A lightweight CNN for facial expression recognition.
    
    Architecture:
    Input: 1x48x48
    - Block 1: Conv(1->32, k=3, pad=1) -> ReLU -> MaxPool(2x2) -> 32x24x24
    - Block 2: Conv(32->64, k=3, pad=1) -> ReLU -> MaxPool(2x2) -> 64x12x12
    - Block 3: Conv(64->128, k=3, pad=1) -> ReLU -> MaxPool(2x2) -> 128x6x6
    - Global Average Pooling -> 128
    - Dropout(p=0.30)
    - Linear(128 -> 7)
    """
    def __init__(self, num_classes: int = 7):
        super(LightweightCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        # Converts 128x6x6 to 128x1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(in_features=128, out_features=num_classes)
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
        
        x = self.global_avg_pool(x)
        
        # Flatten the tensor from (B, 128, 1, 1) to (B, 128)
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)
        
        return x
