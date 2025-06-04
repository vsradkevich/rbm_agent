# src/agent.py

import torch
import numpy as np


class ReversibleAgent:
    """
    Агрегирует архитектуру: RevNet (feature extractor), RBM (energy layer), Critic (free-energy gradient).
    Принимает наблюдение, возвращает действие (policy можно заменить).
    """

    def __init__(self, revnet, rbm, critic, device='cuda'):
        self.revnet = revnet
        self.rbm = rbm
        self.critic = critic
        self.device = device

    def preprocess_obs(self, obs):
        """
        Преобразует наблюдение среды (numpy array) в тензор для подачи в RevNet.
        Приводит к (1, 3, H, W) и нормализует (если нужно).
        Ожидается obs: (H, W, 3) float32, значения 0.0-1.0
        """
        obs = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        obs = obs.float().to(self.device)
        return obs

    def extract_features(self, obs):
        """
        Универсальный forward через RevNet: гарантирует eval режим при batch=1 (BatchNorm-safe).
        """
        obs_tensor = self.preprocess_obs(obs)
        was_training = self.revnet.training
        self.revnet.eval()
        with torch.no_grad():
            features = self.revnet(obs_tensor)
        if was_training:
            self.revnet.train()
        return features

    def act(self, obs, explore=True):
        """
        Делает forward pass: извлекает признаки -> подаёт в RBM -> использует политику (baseline: случайно)
        Можно расширить: выбирать действие с max grad(∂F/∂state), value-based, softmax и т.д.
        """
        _ = self.extract_features(obs)
        # --- Политика выбора действия ---
        if explore:
            # Режим исследования/обучения: случайное действие
            action = np.random.randint(0, 4)
        else:
            # Здесь можно реализовать argmax по value, или grad-based политику
            action = np.random.randint(0, 4)
        return action

    def evaluate_state(self, obs):
        """
        Возвращает признаки, free energy, grad(∂F/∂state) — удобно для стратегии и анализа.
        """
        obs_tensor = self.preprocess_obs(obs).requires_grad_(True)
        was_training = self.revnet.training
        self.revnet.eval()
        features = self.revnet(obs_tensor)
        if was_training:
            self.revnet.train()
        free_energy = self.rbm(features)
        grad = self.critic.evaluate(self.rbm, features)
        return features.detach().cpu().numpy(), free_energy.item(), grad.detach().cpu().numpy()

    def to(self, device):
        """
        Переносит все параметры агента на указанный девайс.
        """
        self.device = device
        self.revnet.to(device)
        self.rbm.to(device)
        # Critic stateless, device-agnostic
