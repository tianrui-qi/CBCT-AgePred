import torch
import torch.cuda
import torch.backends.cudnn

import numpy as np
import random
import os

import cfg, src

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    """Set the seed for reproducibility in PyTorch for multi-GPU."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed

    # If you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensuring that all the GPUs have the same seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    config = cfg.Config()
    trainer = src.Trainer(
        dataset=src.Patients(**config.Patients), 
        model=src.MAE(encoder=src.ViT(**config.ViT), **config.MAE),
        **config.Trainer
    ).fit()
