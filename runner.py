import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.tensorboard.writer as writer
from torch.utils.data import Dataset, DataLoader, random_split

import os

import tqdm


class Trainer:
    def __init__(
        self, max_epoch: int, accumu_steps: int, evalu_frequency: int,
        ckpt_save_fold: str, ckpt_load_path: str, ckpt_load_lr: bool,
        dataset: Dataset, train_num: int, batch_size: int, num_workers: int,
        model: nn.Module, lr: float, gamma: float
    ) -> None:
        self.device = "cuda"
        self.max_epoch = max_epoch
        self.accumu_steps = accumu_steps
        self.evalu_frequency = evalu_frequency
        # path
        self.ckpt_save_fold = ckpt_save_fold
        self.ckpt_load_path = ckpt_load_path
        self.ckpt_load_lr   = ckpt_load_lr

        # dataset
        trainset, evaluset = random_split(
            dataset, [train_num, len(dataset) - train_num]
        )
        trainset.dataset.train()
        evaluset.dataset.evalu()
        # dataloader
        self.trainloader = DataLoader(
            trainset, pin_memory=True, 
            batch_size=batch_size, num_workers=num_workers, 
        )
        self.evaluloader = DataLoader(
            evaluset, pin_memory=True,
            batch_size=batch_size, num_workers=num_workers, 
        )
        # model
        self.model = model
        if torch.cuda.device_count() > 1:
            print(f"Use {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        model.to("cuda")
        # loss
        self.loss = nn.MSELoss()
        # optimizer
        self.scaler    = amp.GradScaler()  # type: ignore
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=gamma
        )
        # recorder
        self.writer = writer.SummaryWriter()

        # index
        self.epoch = 1  # epoch index may update in load_ckpt()

        # print model info
        para_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {para_num:,} trainable parameters')

    def fit(self) -> None:
        self._load_ckpt()
        for self.epoch in tqdm.tqdm(
            range(self.epoch, self.max_epoch+1), 
            total=self.max_epoch, desc=self.ckpt_save_fold, smoothing=0.0,
            unit="epoch", initial=self.epoch
        ):
            self._train_epoch()
            if self.epoch % self.evalu_frequency != 0: continue
            self._valid_epoch()
            self._update_lr()
            self._save_ckpt()

    def _train_epoch(self) -> None:
        self.model.train()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.trainloader)/self.accumu_steps), 
            desc='train_epoch', leave=False, unit="steps", smoothing=1.0
        )
        # record: tensorboard
        train_loss = []

        for i, (frames, labels) in enumerate(self.trainloader):
            # put frames and labels in GPU
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            # forward and backward
            with amp.autocast(dtype=torch.float16):
                predis = self.model(frames)
                loss_value = self.loss(predis, labels) / self.accumu_steps
            self.scaler.scale(loss_value).backward()

            # record: tensorboard
            train_loss.append(loss_value.item() / len(predis))

            # update model parameters
            if (i+1) % self.accumu_steps != 0: continue
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # record: tensorboard
            self.writer.add_scalars(
                'scalars/loss', {'train': torch.sum(torch.as_tensor(train_loss))}, 
                (self.epoch - 1) * len(self.trainloader) / self.accumu_steps + 
                (i + 1) / self.accumu_steps
            )  # average loss of each frame
            train_loss = []
            # record: progress bar
            pbar.update()

    @torch.no_grad()
    def _valid_epoch(self) -> None:
        self.model.eval()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.evaluloader)/self.accumu_steps), 
            desc="valid_epoch", leave=False, unit="steps", smoothing=1.0
        )
        # record: tensorboard
        valid_loss = []

        for i, (frames, labels) in enumerate(self.evaluloader):
            # put frames and labels in GPU
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            # forward
            predis = self.model(frames)
            # loss
            loss_value = self.loss(predis, labels)

            # record: tensorboard
            valid_loss.append(loss_value.item() / len(predis))
            # record: progress bar
            if (i+1) % self.accumu_steps == 0: pbar.update()
        
        # record: tensorboard
        self.writer.add_scalars(
            'scalars/loss', {'valid': torch.mean(torch.as_tensor(valid_loss))}, 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def _update_lr(self) -> None:
        # update learning rate
        if self.scheduler.get_last_lr()[0] > 1e-8: self.scheduler.step()

        # record: tensorboard
        self.writer.add_scalar(
            'scalars/lr', self.optimizer.param_groups[0]['lr'], 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def _save_ckpt(self) -> None:
        # file path checking
        if not os.path.exists(self.ckpt_save_fold): 
            os.makedirs(self.ckpt_save_fold)

        torch.save({
            'epoch': self.epoch,  # epoch index start from 1
            'model': self.model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }, "{}/{}.ckpt".format(self.ckpt_save_fold, self.epoch)
        )

    @torch.no_grad()
    def _load_ckpt(self) -> None:
        if self.ckpt_load_path == "": return
        ckpt = torch.load("{}.ckpt".format(self.ckpt_load_path))
        
        self.epoch = ckpt['epoch']+1  # start train from next epoch index
        self.model.load_state_dict(ckpt['model'])
        self.scaler.load_state_dict(ckpt['scaler'])
        
        if not self.ckpt_load_lr: return
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
