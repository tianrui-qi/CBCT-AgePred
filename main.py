import torch
import lightning as L

import argparse

import src


__all__ = []


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    L.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.add_parser("unet")
    subparsers.add_parser("pretrain")
    subparsers.add_parser("finetune")
    args = parser.parse_args()

    if args.mode == "unet":
        config = src.UNetConfig()
        src.Trainer(
            **config.trainer, 
            trainset=src.UNetDataset(**config.trainset), 
            validset=src.UNetDataset(**config.validset), 
            model=src.UNetModel(**config.model)
        ).fit()
    if args.mode == "pretrain":
        config = src.PretrainConfig()
        src.Trainer(
            **config.trainer, 
            trainset=src.PretrainDataset(**config.trainset), 
            validset=src.PretrainDataset(**config.validset), 
            model=src.PretrainModel(**config.model)
        ).fit()
    if args.mode == "finetune":
        config = src.FinetuneConfig()
        src.Trainer(
            **config.trainer, 
            trainset=src.FinetuneDataset(**config.trainset), 
            validset=src.FinetuneDataset(**config.validset), 
            model=src.FinetuneModel(**config.model)
        ).fit()
