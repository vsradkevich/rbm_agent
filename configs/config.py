# src/config.py

from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    grid_size: int = 8
    buffer_size: int = 65536
    epochs: int = 100
    steps_per_epoch: int = 128
    batch_size: int = 64
    lr_rbm: float = 1e-3
    hidden_dim: int = 32
    noise_prob: float = 0.1
    save_path: str = "checkpoint.pth"
    log_path: str = "logs/"
    render_traj: bool = True
    episodes_inference: int = 3


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
