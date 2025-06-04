# tests/test_rbm_reconstruction.py

import torch
import numpy as np
import sys
import os

# Импортируем RBM из src/model/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/model')))
from rbm import RBM


def reconstruction_score(rbm, x):
    prob_h, h = rbm.sample_hidden(x)
    prob_v, x_recon = rbm.sample_visible(h)
    mse = ((x - prob_v) ** 2).mean().item()
    acc = (x.round() == prob_v.round()).float().mean().item()
    return mse, acc


def test_rbm_reconstruction():
    torch.manual_seed(42)
    batch_size = 32
    visible_dim = 32
    hidden_dim = 8
    epochs = 200

    rbm = RBM(visible_dim=visible_dim, hidden_dim=hidden_dim, lr=1e-2)
    data = torch.bernoulli(torch.rand(batch_size, visible_dim))

    # До обучения
    mse0, acc0 = reconstruction_score(rbm, data)
    print(f"До обучения: MSE = {mse0:.4f}, Accuracy = {acc0:.4f}")

    # Обучение
    for epoch in range(epochs):
        rbm.contrastive_divergence(data)

    # После обучения
    mse1, acc1 = reconstruction_score(rbm, data)
    print(f"После {epochs} шагов CD: MSE = {mse1:.4f}, Accuracy = {acc1:.4f}")

    assert acc1 > acc0 + 0.1, "RBM не учится реконструировать!"


if __name__ == "__main__":
    test_rbm_reconstruction()
    print("✅ RBM reconstruction learning test passed!")
