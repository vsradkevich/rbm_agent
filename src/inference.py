# src/inference.py

import torch
import numpy as np
from agent import ReversibleAgent
from model.tinyrevnet import TinyReversibleConvNet
from model.rbm import RBM
from model.critic import Critic
from env import GridWorld

import matplotlib

matplotlib.use("Agg")  # для WSL/серверов без GUI

import matplotlib.pyplot as plt


def visualize_trajectory(grid_size, traj, goal_pos, episode=None, reward=None, save_dir="logs/trajectories"):
    """
    Визуализация траектории агента. Сохраняет картинку при достижении цели.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    grid = np.zeros((grid_size, grid_size))
    for (x, y) in traj:
        grid[x, y] += 1
    gx, gy = goal_pos
    grid[gx, gy] = np.nan  # goal отображается иначе
    plt.imshow(grid, cmap='Blues')
    plt.title(f'Agent trajectory (white = goal), episode {episode + 1}')
    plt.colorbar()
    plt.scatter([gy], [gx], c='red', marker='*', s=200, label='Goal')
    plt.legend()
    # Если заданы номер эпизода/награда — пишем это в имя файла
    if episode is not None and reward is not None:
        filename = f"ep{episode:03d}_reward{reward:.2f}.png"
    else:
        filename = "trajectory.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def run_lifelong_inference(
        episodes=None,
        grid_size=8,
        device='cuda',
        render_traj=True,
        model_path=None,
        noise_prob=0.1,
        online_train=True,
        verbose=True,
        output_dim=128,
        hidden_dim=16,
):
    # Инициализация среды и агентов
    env = GridWorld(grid_size=grid_size, noise_prob=noise_prob)
    revnet = TinyReversibleConvNet(output_dim=output_dim).to(device)
    rbm = RBM(visible_dim=output_dim, hidden_dim=hidden_dim).to(device)
    critic = Critic()
    agent = ReversibleAgent(revnet, rbm, critic, device=device)

    # Загрузка обученных весов, если есть
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        revnet.load_state_dict(checkpoint['revnet'])
        rbm.load_state_dict(checkpoint['rbm'])
        if verbose:
            print(f"Загружены веса из {model_path}")

    ep = 0
    all_rewards = []
    all_fe = []
    all_steps = []
    try:
        while True:
            if episodes is not None and ep >= episodes:
                break
            state = env.reset()
            done = False
            total_reward = 0
            traj = []
            states_mem = []
            fe_vals = []
            while not done:
                action = agent.act(state, explore=False)
                state_tensor = agent.preprocess_obs(state)
                was_training = revnet.training
                revnet.eval()
                with torch.no_grad():
                    features = revnet(state_tensor)
                    fe = rbm.free_energy(features).cpu().item()
                if was_training:
                    revnet.train()
                fe_vals.append(fe)
                next_state, reward, done = env.step(action)
                total_reward += reward
                states_mem.append(state)
                agent_pos = np.argmax(state[..., 0]) // grid_size, np.argmax(state[..., 0]) % grid_size
                traj.append(agent_pos)
                state = next_state
            # Goal
            goal_pos = np.argmax(state[..., 1]) // grid_size, np.argmax(state[..., 1]) % grid_size
            mean_fe = np.mean(fe_vals) if fe_vals else float("nan")
            all_rewards.append(total_reward)
            all_fe.append(mean_fe)
            all_steps.append(len(traj))
            if verbose:
                print(f"Эпизод {ep + 1}: суммарная награда {total_reward:.2f}, шагов: {len(traj)}, ср. free energy: {mean_fe:.4f}")
            # === Save trajectory only for successful episode ===
            if render_traj and total_reward > 0:
                visualize_trajectory(grid_size, traj, goal_pos, episode=ep + 1, reward=total_reward)
            # === ONLINE TRAIN ===
            if online_train and len(states_mem) > 1:
                # Преобразуем весь опыт эпизода в фичи RevNet и обучаем RBM
                states_batch = np.stack(states_mem).astype(np.float32)  # (steps, H, W, C)
                features_batch = []
                for s in states_batch:
                    st = agent.preprocess_obs(s)
                    with torch.no_grad():
                        f = revnet(st)
                    features_batch.append(f.cpu())
                features_batch = torch.cat(features_batch, dim=0).to(device)
                rbm.contrastive_divergence(features_batch)
            ep += 1
        print("\n== Результаты инференса ==")
        for i, (r, fe, n) in enumerate(zip(all_rewards, all_fe, all_steps)):
            print(f"Эпизод {i + 1}: reward={r:.2f}, steps={n}, mean FE={fe:.4f}")
    except KeyboardInterrupt:
        print("Вечная жизнь агента остановлена пользователем.")


if __name__ == "__main__":
    run_lifelong_inference(
        grid_size=8,
        device='cuda',
        render_traj=True,
        model_path=None,  # или путь к модели, если есть
        online_train=True,
        verbose=True,
        output_dim=128,
        hidden_dim=16
    )
