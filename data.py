import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor
import torchio as tio

import tifffile     # read and save tiff file
import os
import re
import tqdm         # progress bar
from typing import List, Tuple


class Patients(Dataset):
    def __init__(
        self, tiff_fold: str, info_fold: str, 
        min_HU: int, max_HU: int, degrees: float, translation: int, std: float
    ) -> None:
        super(Patients, self).__init__()
        # path
        self.tiff_fold = tiff_fold
        self.info_fold = info_fold
        # normalization
        self.min_HU = min_HU
        self.max_HU = max_HU
        # augmentation
        self.augmentation = None  # control by train() and evalu() mode switch
        self.transform = tio.transforms.Compose([
            tio.RandomFlip(axes=(2,), flip_probability=0.5),
            tio.RandomAffine(
                scales=0, 
                degrees=degrees, 
                translation=translation,
                isotropic=True
            ),
            tio.RandomNoise(std=std),
        ])

        # cache
        self.name_list = [
            os.path.splitext(file)[0] for file in os.listdir(self.tiff_fold)
        ]
        self.info_list = self._getInfoList()

    def _getInfoList(self) -> Tensor:
        info_list = [None] * len(self.name_list)
        for i in tqdm.tqdm(
            range(len(self.name_list)), desc="_getInfoList", leave=False
        ):
            info_path = os.path.join(self.info_fold, self.name_list[i] + ".txt")
            with open(info_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # split
                    split: List[str] = line.strip().split()
                    if len(split) < 2: continue
                    # age
                    y = re.search(r'(\d+)年', split[1])
                    m = re.search(r'(\d+)月', split[1])
                    d = re.search(r'(\d+)天', split[1])
                    y = int(y.group(1)) if y else 0
                    m = int(m.group(1)) if m else 0
                    d = int(d.group(1)) if d else 0
                    age: float = (y * 365.25 + m * 30.5 + d) / 365.25
                    if info_list[i] is None: 
                        info_list[i] = age
                    if info_list[i] != age: print(info_path, "age miss match")
        return torch.tensor(info_list).float()

    def train(self) -> None: 
        self.augmentation = True

    def evalu(self) -> None: 
        self.augmentation = False

    def __getitem__(self, index: int) -> Tuple[Tensor, float]:
        # read the tiff
        tiff = torch.from_numpy(tifffile.imread(
            os.path.join(self.tiff_fold, self.name_list[index] + ".tif")
        )).float()

        # normalization
        tiff = (tiff - self.min_HU) / (self.max_HU - self.min_HU)

        # augmentation (rotation and translation)
        if self.augmentation: 
            tiff = self.transform(tiff.unsqueeze(0)).squeeze(0)

        # crop to (416 416 416)
        if tiff.shape not in ((440, 536, 536), (528, 640, 640), (576, 768, 768)):
            raise ValueError(
                "expect (440, 536, 536), (528, 640, 640), or (576, 768, 768) " +
                "but got " + str(tiff.shape)
            )
        elif tiff.shape == (440, 536, 536):
            tiff = tiff[440-416:440,  50: 50+416,  60: 60+416]
        elif tiff.shape == (528, 640, 640):
            tiff = tiff[528-416:528, 100:100+416, 112:112+416]
        elif tiff.shape == (576, 768, 768):
            tiff = tiff[576-416:576,  50: 50+416, 176:176+416]

        # reshape to (1, 416, 416, 416), i.e., (B, D, H, W)
        tiff = tiff.reshape(1, 416, 416, 416)

        return torch.clip(tiff, 0, 1), self.info_list[index]

    def __len__(self) -> int:
        return len(self.name_list)
