# src/model/revnet.py

import torch
import torch.nn as nn
import torchvision.models as models


class RevNet18(nn.Module):
    """
    Простой обёртка над ResNet-18 (можно заменить на настоящую RevNet).
    Извлекает признаки фиксированной длины для подачи в RBM.
    """

    def __init__(self, output_dim=512):
        super().__init__()
        backbone = models.resnet18(weights=None)  # можно weights='IMAGENET1K_V1' для предобучения
        # Удаляем последний fully-connected слой
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()
        self.output_dim = output_dim

    def forward(self, x):
        """
        x: (batch, 3, H, W)
        Выход: (batch, output_dim)
        """
        features = self.feature_extractor(x)
        features = self.flatten(features)  # (batch, 512)
        return features
