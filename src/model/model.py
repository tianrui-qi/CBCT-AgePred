import torch.optim
import torch.optim.lr_scheduler
import torch.nn.functional as F
from torch import Tensor
import lightning as L

from .vit3d import ViT3D
from .mae import MAE
from .unet import UNet


__all__ = ["UNetModel", "PretrainModel", "FinetuneModel"]


class UNetModel(L.LightningModule):
    def __init__(self, lr: float, unet_kwargs: dict[str, any]) -> None:
        super().__init__()
        self.lr = lr
        self.unet_kwargs = unet_kwargs
        # model
        self.unet = UNet(**unet_kwargs)

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        frame, age = batch
        loss = F.mse_loss(self.unet(frame.unsqueeze(1)).squeeze(1), age)
        self.log("train", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        frame, age = batch
        loss = F.mse_loss(self.unet(frame.unsqueeze(1)).squeeze(1), age)
        self.log("valid", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
        return [optimizer], [scheduler]


class PretrainModel(L.LightningModule):
    def __init__(
        self, lr: float, vit_kwargs: dict[str, any], mae_kwargs: dict[str, any],
    ) -> None:
        super().__init__()
        self.lr = lr
        self.vit_kwargs = vit_kwargs
        self.mae_kwargs = mae_kwargs
        # model
        self.vit = ViT3D(**vit_kwargs)
        self.mae = MAE(encoder=self.vit, **mae_kwargs)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch = batch.unsqueeze(1)  # add channel dimension
        loss = self.mae(batch)
        self.log("train", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch = batch.unsqueeze(1)  # add channel dimension
        loss = self.mae(batch)
        self.log("valid", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
        return [optimizer], [scheduler]


class FinetuneModel(L.LightningModule):
    def __init__(self, lr: float, unet_kwargs: dict[str, any]) -> None:
        super().__init__()
        self.lr = lr
        self.unet_kwargs = unet_kwargs
        # model
        self.unet = UNet(**unet_kwargs)

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        frame, age = batch
        loss = F.mse_loss(self.unet(frame.unsqueeze(1)).squeeze(1), age)
        self.log("train", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        frame, age = batch
        loss = F.mse_loss(self.unet(frame.unsqueeze(1)).squeeze(1), age)
        self.log("valid", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95
        )
        return [optimizer], [scheduler]
