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


class Patient(Dataset):
    def __init__(self) -> None:
        super(Patient, self).__init__()

        self.info_fold = "D:/tao_ct/tao_ct_info"
        self.cbct_fold = "D:/tao_ct/tao_ct_desens"
        self.tiff_fold = "D:/tao_ct/tao_ct_tiff"

        # normalization
        self.normalization = True
        self.min_HU = -1000
        self.max_HU =  5000

        # augmentation
        self.augmentation = True
        self.transform = tio.transforms.Compose([
            tio.RandomFlip(axes=(2,), flip_probability=0.5),
            tio.RandomAffine(
                scales=0,
                degrees=(-15, 15),
                translation=(-20, 20),
                isotropic=True
            ),
            tio.RandomNoise(std=(0, 0.01)),
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
                del shape_num[key]

        return name_list

    def _getInfoList(self) -> List[float]:
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
        return info_list

    def __getitem__(self, index: int) -> Tuple[Tensor, float]:
        # read the tiff
        tiff = torch.from_numpy(tifffile.imread(
            os.path.join(self.tiff_fold, self.name_list[index] + ".tif")
        )).float()

        # normalization
        if self.normalization:    
            tiff = (tiff - self.min_HU) / (self.max_HU - self.min_HU)

        # augmentation (rotation and translation)
        if self.augmentation:
            tiff = self.transform(tiff.unsqueeze(0)).squeeze(0)

        # crop or pad to (480 480 480)
        if tiff.shape != (440, 536, 536) and tiff.shape != (576, 768, 768):
            raise ValueError(
                "expect (440, 536, 536) or (576, 768, 768) but got " + 
                str(tiff.shape)
            )
        elif tiff.shape == (440, 536, 536):
            # Z : pad from 440 to 480
            tiff = F.pad(tiff, (0, 0, 0, 0, 20, 20))
            # HW: crop a (480 480) center region from (536, 536)
            tiff = tiff[:, 28 : 28+480, 28 : 28+480]
        elif tiff.shape == (576, 768, 768):
            tiff = tiff[48:48+480, 50:50+480, 144:144+480]

        return torch.clip(tiff, 0, 1), self.info_list[index]

    def __len__(self) -> int:
        return len(self.name_list)
