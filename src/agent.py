# src/agent.py

import torch
import numpy as np


class ReversibleAgent:
    """
    Агрегирует всю архитектуру: RevNet (feature extractor), RBM (energy layer), Critic (free-energy gradient).
    Принимает наблюдение, возвращает действие (пока — случайное, policy легко заменить).
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
        """
        # Ожидается obs: (H, W, 3) float32, значения 0.0-1.0
        obs = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        obs = obs.float().to(self.device)
        return obs

    def act(self, obs, explore=True):
        """
        Делает forward pass: извлекает признаки -> подаёт в RBM -> использует политику (пока случайно)
        Можно расширить: выбирать действие с максимальным grad(∂F/∂state), value-based policy, softmax и т.д.
        """
        obs_tensor = self.preprocess_obs(obs)
        # --- Исправление: безопасный проход через RevNet при batch=1 ---
        was_training = self.revnet.training
        self.revnet.eval()
        with torch.no_grad():
            features = self.revnet(obs_tensor)
        if was_training:
            self.revnet.train()
        # -------------------------------------------------------------
        # Для PoC действия — случайные (как baseline)
        if explore:
            # Режим обучения/исследования: случайное действие
            action = np.random.randint(0, 4)
        else:
            # Например, argmax по value (можно расширить)
            action = np.random.randint(0, 4)
        return action

    def evaluate_state(self, obs):
        """
        Возвращает признаки и free energy, grad(∂F/∂state)
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
        self.device = device
        self.revnet.to(device)
        self.rbm.to(device)
        # Critic stateless, device-agnostic
