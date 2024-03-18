import torch
import lightning as L

import src


__all__ = []


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    L.seed_everything(42, workers=True)

    config = src.PretrainConfig()
    src.PretrainTrainer(
        **config.trainer, 
        trainset=src.PretrainDataset(**config.trainset), 
        validset=src.PretrainDataset(**config.validset), 
        model=src.PretrainModel(**config.model)
    ).fit()
