# src/model/critic.py

import torch


class Critic:
    """
    Критик: оценивает ∂F/∂state для заданного состояния и RBM.
    Используется для фильтрации сигналов и принятия стратегических решений агентом.
    """

    def __init__(self):
        pass  # Stateless

    def evaluate(self, rbm, state):
        """
        rbm: экземпляр RBM (nn.Module)
        state: torch tensor, (1, feature_dim), requires_grad=True

        Возвращает: numpy вектор градиента ∂F/∂state той же размерности, что и state.
        """
        state = state.clone().detach().requires_grad_(True)
        free_energy = rbm(state)  # shape: [1]
        free_energy.backward()
        grad = state.grad.detach().cpu().numpy()
        state.grad.zero_()
        return grad
