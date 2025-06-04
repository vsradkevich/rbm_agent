# src/inference.py

import torch
import numpy as np
from agent import ReversibleAgent
from model.revnet import RevNet18
from model.rbm import RBM
from model.critic import Critic
from model.compressor import FlowCompressor
from env import GridWorld
import matplotlib.pyplot as plt


def visualize_trajectory(grid_size, traj, goal_pos):
    """
    Быстрая визуализация одной траектории агента в среде.
    """
    grid = np.zeros((grid_size, grid_size))
    for (x, y) in traj:
        grid[x, y] += 1
    gx, gy = goal_pos
    grid[gx, gy] = np.nan  # goal отображается иначе
    plt.imshow(grid, cmap='Blues')
    plt.title('Agent trajectory (white = goal)')
    plt.colorbar()
    plt.scatter([gy], [gx], c='red', marker='*', s=200, label='Goal')
    plt.legend()
    plt.show()


def run_inference(
        episodes=3,
        grid_size=8,
        device='cuda',
        render_traj=True,
        model_path=None,
        noise_prob=0.1,
):
    # Инициализация среды и агентов
    env = GridWorld(grid_size=grid_size, noise_prob=noise_prob)
    revnet = RevNet18().to(device)
    rbm = RBM(visible_dim=512, hidden_dim=32).to(device)
    critic = Critic()
    agent = ReversibleAgent(revnet, rbm, critic, device=device)

    # Загрузка обученных весов, если есть
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        revnet.load_state_dict(checkpoint['revnet'])
        rbm.load_state_dict(checkpoint['rbm'])

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        traj = []
        while not done:
            action = agent.act(state, explore=False)
            next_state, reward, done = env.step(action)
            total_reward += reward
            # Получаем позицию агента для отрисовки
            agent_pos = np.argmax(state[..., 0]) // grid_size, np.argmax(state[..., 0]) % grid_size
            traj.append(agent_pos)
            state = next_state
        # Goal
        goal_pos = np.argmax(state[..., 1]) // grid_size, np.argmax(state[..., 1]) % grid_size
        print(f"Эпизод {ep + 1}: суммарная награда {total_reward:.2f}, шагов: {len(traj)}")
        if render_traj:
            visualize_trajectory(grid_size, traj, goal_pos)


if __name__ == "__main__":
    run_inference(episodes=3, grid_size=8, device='cuda', render_traj=True, model_path=None)
