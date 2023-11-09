import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import Tensor
import torchio as tio

import numpy as np
import os
import re
import pydicom      # read dicom file
import tifffile     # read and save tiff file

import tqdm         # progress bar
from typing import List, Tuple


class Patients(Dataset):
    def __init__(
        self, info_fold: str, cbct_fold: str, tiff_fold: str,
        min_HU: int, max_HU: int,
        degrees: float, translation: int, std: float
    ) -> None:
        super(Patients, self).__init__()
        # path
        self.info_fold = info_fold
        self.cbct_fold = cbct_fold
        self.tiff_fold = tiff_fold
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

        self.name_list = self._getNameList()
        self.info_list = self._getInfoList()

    def _getNameList(self) -> List[str]:
        name_list = os.listdir(self.cbct_fold)

        if not os.path.exists(self.tiff_fold): os.mkdir(self.tiff_fold)

        shape_num = {}
        for i in tqdm.tqdm(range(len(name_list)), desc="_getNameList"):
            # path for current patient
            tiff_path = os.path.join(self.tiff_fold, name_list[i] + ".tif")
            cbct_path = os.path.join(self.cbct_fold, name_list[i])
            if os.path.exists(tiff_path): 
                # if tiff file exists, i.e., we already run thi initialization
                # before, then we just read the tiff file
                frame = tifffile.imread(tiff_path)
            else:
                # read from cbct_fold layer by layer and concat
                frame = []
                file_list = os.listdir(cbct_path)
                for file in file_list:
                    if not file.startswith("CT"): continue
                    dcm_file = pydicom.dcmread(
                        os.path.join(cbct_path, file)
                    )
                    frame.append(dcm_file.pixel_array)
                frame = np.stack(frame)
                # save as tiff file
                tifffile.imwrite(tiff_path, frame)

            # save shape for further use
            if frame.shape not in shape_num: 
                shape_num[frame.shape] = []
            shape_num[frame.shape].append(name_list[i])

        #for key in shape: 
        #    if len(shape[key]) < 50: print(key, shape[key])
        #    else: print(key, len(shape[key]))

        # use the shape we saved to remove the patients with different shape
        for key in shape_num:
            if len(shape_num[key]) < 50:
                for value in shape_num[key]:
                    name_list.remove(value)

        return name_list

    def _getInfoList(self) -> Tensor:
        info_list = [None] * len(self.name_list)
        for i in tqdm.tqdm(range(len(self.name_list)), desc="_getInfoList"):
            info_path = os.path.join(self.info_fold, self.name_list[i] + ".txt")
            with open(info_path, 'r') as file:
                for line in file:
                    # split
                    split: List[str] = line.strip().split()
                    # age
                    y = re.search(r'(\d+)年', split[1])
                    m = re.search(r'(\d+)月', split[1])
                    d = re.search(r'(\d+)天', split[1])
                    y = int(y.group(1)) if y else 0
                    m = int(m.group(1)) if m else 0
                    d = int(d.group(1)) if d else 0
                    age: float = (y * 365.25 + m * 30.5 + d) / 365.25
                    if info_list[i] is None: info_list[i] = age
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
        if tiff.shape != (440, 536, 536) and tiff.shape != (576, 768, 768):
            raise ValueError(
                "expect (440, 536, 536) or (576, 768, 768) but got " + 
                str(tiff.shape)
            )
        elif tiff.shape == (440, 536, 536):
            tiff = tiff[440-416:440,  0:   416,  60: 60+416]
        elif tiff.shape == (576, 768, 768):
            tiff = tiff[576-416:576, 50:50+416, 176:176+416]

        return torch.clip(tiff, 0, 1), self.info_list[index]

    def __len__(self) -> int:
        return len(self.name_list)
