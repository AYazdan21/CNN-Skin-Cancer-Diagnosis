"""
src/model.py
Architectures for Skin Lesion Classification:
1. SkinCancerCNN: Optimized 4-block CNN architecture from scratch matching Table 2 & Fig. 4 (132,583 params).
2. EfficientNet-B0: State-of-the-art transfer learning backbone pre-trained on ImageNet (~5.3M params).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.config import NUM_CLASSES, DROPOUT_RATE


class SkinCancerCNN(nn.Module):
    """
    Optimized 4-block CNN architecture from the paper with:
    - Conv2D (16, 3x3) -> [BN] -> MaxPool -> Conv2D (32, 3x3) -> [BN] -> MaxPool
    - Conv2D (64, 3x3) -> [BN] -> MaxPool (ceil) -> Conv2D (128, 3x3) -> [BN] -> MaxPool
    - AdaptiveAvgPool2d((2, 2)) -> Flatten (512) -> Dense (64) -> Dropout -> Dense (32) -> Dense (7)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = DROPOUT_RATE,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        # Conv Block 1: 3 channels -> 16 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 16 channels -> 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: 32 channels -> 64 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Conv Block 4: 64 channels -> 128 channels
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pool ensures (2, 2) spatial output regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

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
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))

        # Adaptive spatial pooling
        x = self.adaptive_pool(x)

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


def build_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Builds EfficientNet-B0 with ImageNet pre-trained weights and a custom classification head.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # In EfficientNet-B0, classifier is Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )

    # Helper method for consistent API
    def count_parameters():
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.count_parameters = count_parameters
    return model


def build_model(
    model_name: str = "custom_cnn",
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    use_batch_norm: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """
    Factory function:
    - 'custom_cnn': The paper's lightweight CNN built from scratch (132k params)
    - 'efficientnet': Pre-trained EfficientNet-B0 for high-accuracy transfer learning (~5.3M params)
    """
    name = model_name.lower().replace("-", "_")
    if "efficientnet" in name:
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    else:
        return SkinCancerCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
