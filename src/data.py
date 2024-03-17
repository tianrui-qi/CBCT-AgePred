import torch
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
import torchio as tio

import pandas as pd

import os
import re
import tqdm
import tifffile


__all__ = ["PretrainDataset", "FinetuneDataset"]


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self, num: int, num_sampling: int, dim: list[int],
        tiff_load_fold: str, 
        min_HU: int = -1000, max_HU: int = 5000,
        degrees: float | None = 5.0,
    ) -> None:
        super(PretrainDataset, self).__init__()
        # dim
        self.num = num
        self.num_sampling = num_sampling
        self.dim = dim
        # path
        self.tiff_load_fold = tiff_load_fold
        self.tiff_list = os.listdir(tiff_load_fold)
        # normalization
        self.min_HU = min_HU
        self.max_HU = max_HU
        # augmentation
        self.transform = tio.transforms.Compose([
            tio.RandomFlip(axes=(2,), flip_probability=0.5),
            tio.RandomAffine(scales=0, degrees=degrees, isotropic=True),
        ]) if degrees else None

        # cache
        self.count = self.num_sampling
        self.tiff = None

    def __getitem__(self, index: int) -> Tensor:
        if self.count >= self.num_sampling:
            # read the tiff, random select one
            i = torch.randint(0, len(self.tiff_list), (1,)).item()
            self.tiff = torch.from_numpy(tifffile.imread(os.path.join(
                self.tiff_load_fold, self.tiff_list[i]
            ))).float()
            # normalization
            self.tiff = (self.tiff - self.min_HU) / (self.max_HU - self.min_HU)
            self.tiff = torch.clip(self.tiff, 0, 1)
            # augmentation
            if self.transform != None:
                self.tiff = self.transform(self.tiff.unsqueeze(0)).squeeze(0)
            # reset count
            self.count = 0
            print(index)
        self.count += 1

        # random select a self.dim patch
        D, H, W = self.tiff.shape
        edge = torch.tensor(self.dim) // 2
        start_d = torch.randint(-edge[0], D-edge[0], (1,))
        start_h = torch.randint(-edge[1], H-edge[1], (1,))
        start_w = torch.randint(-edge[2], W-edge[2], (1,))
        patch = torch.zeros(self.dim, dtype=torch.float32)
        patch[
            slice(max(0, -start_d), min(self.dim[0], D-start_d)), 
            slice(max(0, -start_h), min(self.dim[1], H-start_h)), 
            slice(max(0, -start_w), min(self.dim[2], W-start_w)),
        ] = self.tiff[
            slice(max(start_d, 0), min(start_d + self.dim[0], D)), 
            slice(max(start_h, 0), min(start_h + self.dim[1], H)),
            slice(max(start_w, 0), min(start_w + self.dim[2], W)),
        ]

        return patch

    def __len__(self) -> int:
        return self.num


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self, dim: list[int], stride: list[int],
        info_load_fold: str, tiff_load_fold: str, 
        min_HU: int = -1000, max_HU: int = 5000,
    ) -> None:
        super(FinetuneDataset, self).__init__()
        # dim
        self.dim = dim
        self.stride = stride
        # path
        self.info_load_fold = info_load_fold
        self.tiff_load_fold = tiff_load_fold
        # normalization
        self.min_HU = min_HU
        self.max_HU = max_HU

        # profile
        self.profile = self.getProfile(self.info_load_fold, self.tiff_load_fold)

    @staticmethod
    def getProfile(info_load_fold: str, tiff_load_fold: str) -> pd.DataFrame:
        info_list = [
            sample.split('.')[0] 
            for sample in os.listdir(info_load_fold) if sample.endswith(".txt")
        ]
        tiff_list = [
            sample.split('.')[0]
            for sample in os.listdir(tiff_load_fold) if sample.endswith(".tif")
        ]
        sample_list = list(set(info_list) & set(tiff_list))
        sample_list.sort()

        profile = {"sample": sample_list, "age": [], "sex": [], "shape": []}
        for sample in tqdm.tqdm(
            sample_list, smoothing=0.0, unit="sample", desc="getProfile"
        ):
            # path
            info_path = os.path.join(info_load_fold, sample + ".txt")
            tiff_path = os.path.join(tiff_load_fold, sample + ".tif")
            # read info
            with open(info_path, 'r', encoding='utf-8') as file:
                split: list[str] = file.readline().strip().split()
            # age
            y = re.search(r'(\d+)年', split[1])
            m = re.search(r'(\d+)月', split[1])
            d = re.search(r'(\d+)天', split[1])
            y = int(y.group(1)) if y else 0
            m = int(m.group(1)) if m else 0
            d = int(d.group(1)) if d else 0
            profile["age"].append((y * 365.25 + m * 30.5 + d) / 365.25)
            # sex
            profile["sex"].append(int(split[2]))
            # shape
            with tifffile.TiffFile(tiff_path) as tif:
                profile["shape"].append((len(tif.pages), *tif.pages[0].shape))
        profile = pd.DataFrame(profile)

        return profile

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # read the tiff
        tiff = torch.from_numpy(tifffile.imread(os.path.join(
            self.tiff_load_fold, self.profile.loc[index]["sample"] + ".tif"
        ))).float()
        # normalization
        tiff = (tiff - self.min_HU) / (self.max_HU - self.min_HU)
        tiff = torch.clip(tiff, 0, 1)
        # pad to (580, 860, 860)
        pad = (torch.tensor([580, 860, 860]) - torch.tensor(tiff.shape))
        pad = (
            pad[2] // 2, pad[2] - pad[2] // 2,
            pad[1] // 2, pad[1] - pad[1] // 2,
            pad[0] // 2, pad[0] - pad[0] // 2,
        )
        tiff = F.pad(tiff, pad)
        # batch into number of self.dim patch
        patches = []
        for z in range(0, tiff.shape[0]-self.dim[0]+1, self.stride[0]):
            for y in range(0, tiff.shape[1]-self.dim[1]+1, self.stride[1]):
                for x in range(0, tiff.shape[2]-self.dim[2]+1, self.stride[2]):
                    patches.append(tiff[
                        z : z+self.dim[0], 
                        y : y+self.dim[1], 
                        x : x+self.dim[2]
                    ])
        patches = torch.stack(patches)

        # age
        age = torch.tensor(self.profile.loc[index]["age"]).float()

        return patches, age

    def __len__(self) -> int:
        return len(self.profile)
