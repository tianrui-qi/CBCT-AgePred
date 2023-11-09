import torch.backends.cudnn

import numpy as np
import random
import os

import config, data, model, runner

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

    cfg = config.Config()
    # data
    patients = data.Patients(**cfg.data_config)
    # model
    resattnet = model.ResAttNet(**cfg.model_config)
    # runner
    trainer = runner.Trainer(
        dataset=patients, model=resattnet, **cfg.runner_config
    )

    trainer.fit()
