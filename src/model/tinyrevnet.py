# src/model/tinyrevnet.py

import torch
import torch.nn as nn


class RevBlock(nn.Module):
    """
    Элементарный обратимый блок: разрезаем по каналам, часть - через conv, часть - просто прибавляется.
    """

    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)  # разрезаем по каналам
        y1 = x1 + self.f(x2)
        y2 = x2
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        x2 = y2
        x1 = y1 - self.f(x2)
        return torch.cat([x1, x2], dim=1)


class TinyReversibleConvNet(nn.Module):
    """
    Мини-реверсивная сверточная сеть для маленьких GridWorld-изображений.
    Всегда output_dim = channels * H * W // 2, чтобы работал RevBlock.
    """

    def __init__(self, input_channels=3, grid_size=8, output_dim=32, n_blocks=2):
        super().__init__()
        self.init_conv = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.rev_blocks = nn.Sequential(
            *[RevBlock(16) for _ in range(n_blocks)]
        )
        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * grid_size * grid_size, output_dim)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.rev_blocks(x)
        x = self.final(x)
        return x
