# src/env.py
import numpy as np
import random


class GridWorld:
    """
    Простейшая среда: агент на квадратной сетке, цель (goal) ставится случайно.
    Агент получает -0.01 за каждый шаг, +1.0 — за попадание в цель, эпизод завершается при достижении цели или после max_steps.
    Есть функция render() для отрисовки текущей позиции.
    """

    def __init__(self, grid_size=8, max_steps=64, noise_prob=0.1, seed=42):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.noise_prob = noise_prob
        self.rng = np.random.default_rng(seed)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Вправо, влево, вниз, вверх
        self.reset()

    def reset(self):
        self.agent_pos = [self.rng.integers(self.grid_size), self.rng.integers(self.grid_size)]
        while True:
            self.goal_pos = [self.rng.integers(self.grid_size), self.rng.integers(self.grid_size)]
            if self.goal_pos != self.agent_pos:
                break
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        """action — int: 0=вправо, 1=влево, 2=вниз, 3=вверх"""
        dx, dy = self.actions[action]
        # Добавляем шум к движению с вероятностью noise_prob
        if random.random() < self.noise_prob:
            dx, dy = self.actions[self.rng.integers(len(self.actions))]
        new_x = np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1)
        self.agent_pos = [new_x, new_y]
        self.steps += 1

        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps
        reward = 1.0 if self.agent_pos == self.goal_pos else -0.01
        return self._get_obs(), reward, done

    def _get_obs(self):
        """Наблюдение: изображение (grid_size x grid_size x 3), где один цвет — агент, другой — цель"""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1.0  # Агент = красный
        obs[self.goal_pos[0], self.goal_pos[1], 1] = 1.0  # Цель = зелёный
        return obs

    def sample_action(self):
        """Случайное действие"""
        return self.rng.integers(len(self.actions))

    def render(self, mode='rgb_array'):
        """Возвращает numpy-массив (grid_size x grid_size x 3) для визуализации"""
        return self._get_obs()
