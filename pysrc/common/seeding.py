import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """Give each DataLoader worker a deterministic, distinct seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
