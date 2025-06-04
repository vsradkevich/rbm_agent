# src/config.py

from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    grid_size: int = 8
    buffer_size: int = 4096      # По умолчанию - оптимально для небольших dim
    epochs: int = 30             # Для быстрого PoC
    steps_per_epoch: int = 128
    batch_size: int = 16
    output_dim: int = 128        # Размер выходного вектора RevNet/TinyRevNet
    hidden_dim: int = 16         # Размер скрытого слоя RBM
    lr_rbm: float = 1e-3
    noise_prob: float = 0.1
    save_path: str = "checkpoint.pth"
    log_path: str = "logs/"
    render_traj: bool = True
    episodes_inference: int = 10
    online_train: bool = True    # Для inference: учиться во время "жизни"
    verbose: bool = True


def set_global_seed(seed):
    import torch, numpy as np, random, os
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
