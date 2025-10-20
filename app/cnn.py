# app/cnn.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Input: 3x64x64
    Conv(3->16, k=3,s=1,p=1) + ReLU
    MaxPool(2,2)
    Conv(16->32, k=3,s=1,p=1) + ReLU
    MaxPool(2,2)
    Flatten
    FC 100 + ReLU
    FC 10
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 64x64 -> after two 2x2 pools: 16x16, channels=32 => 32*16*16
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
