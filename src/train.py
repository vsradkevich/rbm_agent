# src/train.py

import torch
import numpy as np
import random

from model.tinyrevnet import TinyReversibleConvNet
from model.rbm import RBM
from model.critic import Critic
from model.compressor import FlowCompressor
from buffer import RingBuffer
from agent import ReversibleAgent
from env import GridWorld


def set_seed(seed=42):
    """
    Фиксирует seed для воспроизводимости экспериментов.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
        epochs=100,
        steps_per_epoch=128,
        buffer_size=65536,
        batch_size=64,
        grid_size=8,
        device='cuda',
        lr_rbm=1e-3,
        output_dim=128,  # новый параметр: размер выходных признаков RevNet/RBM
        hidden_dim=32,
        noise_prob=0.1,
        verbose=True,
        save_path=None,
        seed=42,
):
    """
    Главный цикл обучения PoC-агента:
    - Инициализация среды, моделей и буфера.
    - Сбор траекторий, обработка наблюдений через RevNet (eval для batch=1).
    - Обучение RBM по признакам из буфера.
    - Логирование метрик (total_reward, free_energy).
    """
    set_seed(seed)
    env = GridWorld(grid_size=grid_size, noise_prob=noise_prob)
    revnet = TinyReversibleConvNet(output_dim=output_dim).to(device)
    rbm = RBM(visible_dim=output_dim, hidden_dim=hidden_dim, lr=lr_rbm).to(device)
    critic = Critic()
    compressor = FlowCompressor(dim=output_dim).to(device)  # Пока не используется явно
    buffer = RingBuffer(state_dim=output_dim, size=buffer_size)
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

            # CD-обучение RBM: на каждом шаге буфер скользит по свежим признакам
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
    # Можно быстро сменить размерность output_dim и hidden_dim для ускорения
    train(
        epochs=30,
        steps_per_epoch=128,
        grid_size=8,
        device='cuda',
        output_dim=128,  # меньше — быстрее, достаточно для GridWorld
        hidden_dim=16,  # можно уменьшить для легкой экспериментации
        batch_size=16,
        buffer_size=4096,
        seed=42
    )
