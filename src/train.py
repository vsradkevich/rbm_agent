# src/train.py

import torch
from model.revnet import RevNet18
from model.rbm import RBM
from model.critic import Critic
from model.compressor import FlowCompressor
from buffer import RingBuffer
from agent import ReversibleAgent
from env import GridWorld

import numpy as np


def train(
        epochs=100,
        steps_per_epoch=128,
        buffer_size=65536,
        batch_size=64,
        grid_size=8,
        device='cuda',
        lr_rbm=1e-3,
        hidden_dim=32,
        noise_prob=0.1,
        verbose=True,
        save_path=None,
):
    """
    Главный цикл обучения PoC-агента:
    - Инициализация среды, моделей и буфера.
    - Сбор траекторий, обработка наблюдений через RevNet (eval для batch=1).
    - Обучение RBM по признакам из буфера.
    - Логирование метрик (total_reward, free_energy).
    """
    # Инициализация среды и компонентов
    env = GridWorld(grid_size=grid_size, noise_prob=noise_prob)
    revnet = RevNet18().to(device)
    rbm = RBM(visible_dim=512, hidden_dim=hidden_dim, lr=lr_rbm).to(device)
    critic = Critic()
    compressor = FlowCompressor(dim=512).to(device)  # Пока не используется явно
    buffer = RingBuffer(state_dim=512, size=buffer_size)
    agent = ReversibleAgent(revnet, rbm, critic, device=device)

    rewards_log = []
    free_energy_log = []

    for epoch in range(epochs):
        state = env.reset()
        total_reward = 0
        free_energy_vals = []
        for step in range(steps_per_epoch):
            action = agent.act(state, explore=True)
            next_state, reward, done = env.step(action)

            # Преобразование наблюдения для RevNet (batch=1 → eval для избежания ошибки BatchNorm)
            state_tensor = agent.preprocess_obs(state)
            was_training = revnet.training
            revnet.eval()
            with torch.no_grad():
                features = revnet(state_tensor)
            if was_training:
                revnet.train()
            features_np = features.squeeze(0).cpu().numpy()
            buffer.add(features_np, reward)

            # Обучение RBM по батчам из буфера (batch_size >= 2, ошибка BatchNorm не возникает)
            if len(buffer) >= batch_size:
                batch_states, _ = buffer.sample(batch_size)
                batch_states = torch.tensor(batch_states, dtype=torch.float32).to(device)
                rbm.contrastive_divergence(batch_states)
                # Логирование free energy для одного из элементов батча
                with torch.no_grad():
                    fe = rbm.free_energy(batch_states[0:1]).cpu().item()
                    free_energy_vals.append(fe)

            total_reward += reward
            state = next_state
            if done:
                break

        mean_fe = np.mean(free_energy_vals) if free_energy_vals else 0.0
        rewards_log.append(total_reward)
        free_energy_log.append(mean_fe)
        if verbose:
            print(f"Эпоха {epoch + 1}/{epochs} | Суммарная награда: {total_reward:.3f} | Ср. free energy: {mean_fe:.4f}")

    # Сохраняем модель и метрики обучения
    if save_path:
        torch.save({
            'revnet': revnet.state_dict(),
            'rbm': rbm.state_dict(),
            'rewards_log': rewards_log,
            'free_energy_log': free_energy_log
        }, save_path)
    return rewards_log, free_energy_log


if __name__ == "__main__":
    train(epochs=30, steps_per_epoch=128, grid_size=8, device='cuda')
