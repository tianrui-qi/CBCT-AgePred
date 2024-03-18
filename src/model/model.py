import torch.nn.functional as F
import torch.optim
import torch.optim.lr_scheduler
import lightning as L

from .vit3d import ViT3D
from .mae import MAE


__all__ = ["PretrainModel", "FinetuneModel"]


class PretrainModel(L.LightningModule):
    def __init__(
        self, lr: float, T_max: int,
        vit_kwargs: dict[str, any], mae_kwargs: dict[str, any],
    ):
        super().__init__()
        self.lr = lr
        self.T_max = T_max
        # model
        self.vit = ViT3D(**vit_kwargs)
        self.mae = MAE(encoder=self.vit, **mae_kwargs)

    def training_step(self, batch, batch_idx):
        batch = batch.unsqueeze(1)  # add channel dimension
        loss = self.mae(batch)
        self.log(
            "train", loss, 
            on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.unsqueeze(1)  # add channel dimension
        loss = self.mae(batch)
        self.log(
            "valid", loss, 
            on_step=True, on_epoch=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # cheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizer, gamma=0.95
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=1e-10
        )
        return [optimizer], [scheduler]


class FinetuneModel(L.LightningModule):
    def __init__(
        self, lr: float, T_max: int,
        vit_kwargs: dict[str, any], unet_kwargs: dict[str, any],
    ):
        self.lr = lr
        self.T_max = T_max
        # model
        self.vit = ViT3D(**vit_kwargs)
        self.unet = None

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # cheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     self.optimizer, gamma=0.95
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max
        )
        return [optimizer], [scheduler]
