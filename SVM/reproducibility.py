import numpy as np
import torch
import optuna

def set_random_seeds(seed=42):
    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)

    # Set seed for PyTorch on GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set seed for Optuna
    optuna.seed(seed)
