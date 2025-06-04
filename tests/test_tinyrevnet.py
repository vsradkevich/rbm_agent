# tests/test_tinyrevnet.py

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Добавляем путь src/model, чтобы tinyrevnet импортировался корректно
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/model')))

from tinyrevnet import RevBlock


def test_revblock_inverse():
    torch.manual_seed(42)
    batch_size = 2
    channels = 16  # Должно делиться на 2!
    height, width = 8, 8
    x = torch.randn(batch_size, channels, height, width)

    block = RevBlock(dim=channels)
    y = block(x)
    x_rec = block.inverse(y)

    diff = (x - x_rec).abs().max().item()
    print(f"Max absolute difference after inversion: {diff:.6e}")
    assert np.allclose(x.detach().numpy(), x_rec.detach().numpy(), atol=1e-5), "Reversible block inversion failed!"


if __name__ == "__main__":
    test_revblock_inverse()
    print("✅ RevBlock inversion test passed!")
