# src/model/compressor.py

import torch
import torch.nn as nn


class FlowCompressor(nn.Module):
    """
    Заглушка для обратимого (flow-based) компрессора признаков.
    Позволяет позже вставить нормализующие потоки типа NICE/RealNVP.
    Сейчас — тождественное отображение (x -> x).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Здесь может быть цепочка coupling-блоков или invertible layers

    def forward(self, x, reverse=False):
        """
        x: (batch, dim)
        reverse: если True — обратное преобразование (future use)
        Сейчас просто возвращает x как есть.
        """
        return x
