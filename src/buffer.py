# src/buffer.py

import numpy as np


class RingBuffer:
    """
    Кольцевой буфер для хранения признаков (rev-кодов) и наград.
    Позволяет быстро накапливать большой опыт и делать случайную выборку батчей для обучения.
    """

    def __init__(self, state_dim, size=65536):
        self.size = size
        self.ptr = 0
        self.full = False
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)

    def add(self, state, reward):
        """
        Сохраняет новый (state, reward) в буфере.
        state — numpy array или torch tensor (state_dim,)
        reward — float
        """
        self.states[self.ptr] = np.array(state, dtype=np.float32)
        self.rewards[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        """
        Случайная выборка batch_size элементов из буфера.
        Возвращает (batch_states, batch_rewards).
        """
        max_idx = self.size if self.full else self.ptr
        idx = np.random.choice(max_idx, batch_size, replace=False)
        return self.states[idx], self.rewards[idx]

    def __len__(self):
        return self.size if self.full else self.ptr
