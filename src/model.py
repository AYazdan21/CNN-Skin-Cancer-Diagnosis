"""
src/model.py
Optimized CNN architecture for dermatological lesion classification matching Table 2 & Fig. 4 of:
'Enhanced skin cancer diagnosis using optimized CNN architecture and checkpoints for automated dermatological lesion classification'
Total trainable parameters: 132,543
"""

import torch
import torch.nn as nn
from src.config import NUM_CLASSES, DROPOUT_RATE


class SkinCancerCNN(nn.Module):
    """
    Optimized 4-block CNN architecture with:
    - Conv2D (16, 3x3) -> MaxPool -> Conv2D (32, 3x3) -> MaxPool
    - Conv2D (64, 3x3) -> MaxPool (ceil) -> Conv2D (128, 3x3) -> MaxPool
    - Flatten (512) -> Dense (64) -> Dropout -> Dense (32) -> Dense (7)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE):
        super().__init__()

        # Conv Block 1: 3x28x28 -> 16x28x28 -> 16x14x14
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 16x14x14 -> 32x14x14 -> 32x7x7
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: 32x7x7 -> 64x7x7 -> 64x4x4 (ceil_mode preserves 4x4 spatial shape)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Conv Block 4: 64x4x4 -> 128x4x4 -> 128x2x2
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier: 128*2*2 = 512 -> 64 -> 32 -> 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 64)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.relu_fc2 = nn.ReLU()
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(self.relu1(self.conv1(x)))
        # Block 2
        x = self.pool2(self.relu2(self.conv2(x)))
        # Block 3
        x = self.pool3(self.relu3(self.conv3(x)))
        # Block 4
        x = self.pool4(self.relu4(self.conv4(x)))

        # Dense classification head
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu_fc2(self.fc2(x))
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes: int = NUM_CLASSES, dropout_rate: float = DROPOUT_RATE) -> SkinCancerCNN:
    return SkinCancerCNN(num_classes=num_classes, dropout_rate=dropout_rate)
