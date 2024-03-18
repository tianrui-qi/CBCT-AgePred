import torch.utils.data
import lightning as L
import lightning.pytorch.loggers
import lightning.pytorch.callbacks

import os

import src.data, src.model


__all__ = ["PretrainTrainer", "FineturnTrainer"]


class PretrainTrainer:
    def __init__(
        self, max_epoch: int, accumu_steps: int, 
        batch_size: int, num_workers: int,
        version: str | None, save_top_k: int, ckpt_load_path: str | None,
        trainset: src.data.PretrainDataset, 
        validset: src.data.PretrainDataset,
        model: src.model.PretrainModel,
    ) -> None:
        # data
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, num_workers=num_workers, 
            persistent_workers=True, pin_memory=True,
        )
        self.validloader = torch.utils.data.DataLoader(
            validset, batch_size=batch_size, num_workers=num_workers, 
            persistent_workers=True, pin_memory=True,
        )
        # model
        self.model = model
        # recoder
        self.ckpt_load_path = ckpt_load_path
        self.checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=os.path.join("ckpt", version) if version else None, 
            monitor="valid", save_top_k=save_top_k
        )
        self.logger = lightning.pytorch.loggers.TensorBoardLogger(
            "runs", name=None, version=version, default_hp_metric=False
        )
        # trainer
        self.trainer = L.Trainer(
            accumulate_grad_batches=accumu_steps,
            precision="16-mixed",
            benchmark=True,
            max_epochs=max_epoch,
            callbacks=[self.checkpoint],     
            logger=self.logger, 
            log_every_n_steps=1,
        )

    def fit(self) -> None:
        self.trainer.fit(
            self.model, 
            train_dataloaders=self.trainloader, 
            val_dataloaders=self.validloader,
            ckpt_path=self.ckpt_load_path,
        )


class FineturnTrainer:
    def __init__(self) -> None:
        pass

    def fit(self) -> None:
        pass
