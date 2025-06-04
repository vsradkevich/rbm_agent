# src/model/rbm.py

import torch
import torch.nn as nn


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine (RBM)
    - visible_dim: размер входного вектора (например, 512)
    - hidden_dim: размер скрытого слоя (например, 32)
    """

    def __init__(self, visible_dim, hidden_dim=32, k=1, lr=1e-3):
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.k = k  # Число шагов Гиббса (обычно 1, CD-1)
        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def sample_hidden(self, v):
        # Вероятности и образцы скрытых единиц (sigmoid и бернуллиевская выборка)
        prob_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return prob_h, torch.bernoulli(prob_h)

    def sample_visible(self, h):
        prob_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return prob_v, torch.bernoulli(prob_v)

    def free_energy(self, v):
        wx_b = torch.matmul(v, self.W) + self.h_bias  # (batch, hidden)
        h_term = torch.log1p(torch.exp(wx_b)).sum(dim=1)  # log(1+exp(.))
        vbias_term = torch.matmul(v, self.v_bias)
        free_energy = -vbias_term - h_term
        return free_energy

    def forward(self, v):
        # Для совместимости с Critic: просто free energy (размер [batch])
        return self.free_energy(v)

    def contrastive_divergence(self, v0):
        """
        Один шаг CD-1 для батча v0 (batch, visible_dim)
        """
        # Положительная фаза
        prob_h0, h0 = self.sample_hidden(v0)
        # CD-кольцо: k шагов Gibbs sampling
        v_neg = v0
        for _ in range(self.k):
            prob_h, h = self.sample_hidden(v_neg)
            prob_v, v_neg = self.sample_visible(h)
        prob_h_neg, _ = self.sample_hidden(v_neg)
        # Градиенты
        positive_grad = torch.matmul(v0.t(), prob_h0)
        negative_grad = torch.matmul(v_neg.t(), prob_h_neg)
        dW = (positive_grad - negative_grad) / v0.size(0)
        dv_bias = torch.mean(v0 - v_neg, dim=0)
        dh_bias = torch.mean(prob_h0 - prob_h_neg, dim=0)
        # Градиентный шаг вручную
        self.W.grad = -dW
        self.v_bias.grad = -dv_bias
        self.h_bias.grad = -dh_bias
        self.optimizer.step()
        self.optimizer.zero_grad()
