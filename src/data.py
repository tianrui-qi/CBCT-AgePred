import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch import Tensor
import torchio as tio

import pandas as pd
import skimage.util

import os
import re
import tqdm
import tifffile
import collections
from typing import Literal

import src.model


__all__ = [
    "UNetDataset",
    "PretrainDataset", "FinetuneDataset"
]


class UNetDataset(torch.utils.data.Dataset):
    def __init__(
        self, profile_load_path: str,
        min_HU: int = 0, max_HU: int = 4000,
        degrees: float | None = 5.0, translation: int | None = 10,
        mode: str = Literal["whole", "teeth"],
    ) -> None:
        super(UNetDataset, self).__init__()
        # profile
        self.profile = pd.read_csv(profile_load_path)
        # normalization
        self.min_HU = min_HU
        self.max_HU = max_HU
        # augmentation
        self.transform = tio.transforms.Compose([
            tio.RandomFlip(axes=(2,), flip_probability=0.5),
            tio.RandomAffine(
                scales=0, degrees=degrees, 
                translation=translation, isotropic=True
            ),
        ]) if degrees or translation else None
        # mode
        self.mode = mode

    def __getitem__(self, index: int) -> tuple[Tensor, float]:
        # read the tiff
        tiff = tifffile.imread(self.profile["tiff"].iloc[index])
        tiff = torch.from_numpy(tiff).float()
        # normalization
        tiff = (tiff - self.min_HU) / (self.max_HU - self.min_HU)
        tiff = torch.clip(tiff, 0, 1)
        # augmentation (rotation and translation)
        if self.transform != None:
            tiff = self.transform(tiff.unsqueeze(0)).squeeze(0)
        # crop to (416 416 416) for whole or (272 272 320) for teeth
        if self.mode == "whole":
            if tiff.shape == (440, 536, 536):
                tiff = tiff[440-416:440,  50: 50+416,  60: 60+416]
            if tiff.shape == (528, 640, 640):
                tiff = tiff[528-416:528, 100:100+416, 112:112+416]
            if tiff.shape == (576, 768, 768):
                tiff = tiff[576-416:576,  50: 50+416, 176:176+416]
        if self.mode == "teeth":
            if tiff.shape == (440, 536, 536): tiff = tiff[
                138:410, 80:352, tiff.shape[2]//2-160:tiff.shape[2]//2+160
            ]
            if tiff.shape == (528, 640, 640): tiff = tiff[
                206:478, 80:352, tiff.shape[2]//2-160:tiff.shape[2]//2+160
            ]
            if tiff.shape == (576, 768, 768): tiff = tiff[
                244:516, 80:352, tiff.shape[2]//2-160:tiff.shape[2]//2+160
            ]

        # age
        age = torch.tensor(self.profile.loc[index]["age"]).float()

        return tiff, age

    def __len__(self) -> int:
        return len(self.profile)


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self, num_sample: int, num_sampling: int, dim: list[int],
        profile_load_path: str,
        min_HU: int = 0, max_HU: int = 4000,
        degrees: float | None = 5.0,
    ) -> None:
        super(PretrainDataset, self).__init__()
        # dim
        self.num_sample = num_sample
        self.num_sampling = num_sampling
        self.dim = dim
        # profile
        self.profile = pd.read_csv(profile_load_path)
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
            i = torch.randint(0, len(self.profile), (1,)).item()
            self.tiff = tifffile.imread(self.profile["tiff"].iloc[i])
            self.tiff = torch.from_numpy(self.tiff).float()
            # normalization
            self.tiff = (self.tiff - self.min_HU) / (self.max_HU - self.min_HU)
            self.tiff = torch.clip(self.tiff, 0, 1)
            # augmentation
            if self.transform != None:
                self.tiff = self.transform(self.tiff.unsqueeze(0)).squeeze(0)
            # reset count
            self.count = 0
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
        return self.num_sample * self.num_sampling


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self, dim: list[int], stride: list[int],
        profile_load_path: str,
        vit_kwargs: dict[str, any], ckpt_load_path: str,
        min_HU: int = 0, max_HU: int = 4000,
    ) -> None:
        super(FinetuneDataset, self).__init__()
        # dim
        self.dim = dim
        self.stride = stride
        # profile
        self.profile = pd.read_csv(profile_load_path)
        # vit
        self.vit = self._getVit(vit_kwargs, ckpt_load_path)
        # normalization
        self.min_HU = min_HU
        self.max_HU = max_HU

    def _getVit(
        self, vit_kwargs: dict[str, any], ckpt_load_path: str,
    ) -> src.model.ViT3D:
        # format the ckpt file
        ckpt = torch.load(ckpt_load_path)["state_dict"]
        ckpt_vit = collections.OrderedDict()
        for key, value in ckpt.items():
            if key.startswith('vit.'):
                new_key = key[4:]
                ckpt_vit[new_key] = value
        # load the vit
        vit: nn.Module = src.model.ViT3D(**vit_kwargs)
        vit.load_state_dict(ckpt_vit)
        vit.eval()
        for param in vit.parameters(): param.requires_grad = False
        if torch.cuda.is_available(): vit.to("cuda")

        return vit

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        # read the tiff
        tiff = tifffile.imread(self.profile["tiff"].iloc[index])
        tiff = torch.from_numpy(tiff).float()
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
        patches = torch.from_numpy(skimage.util.view_as_windows(
            tiff.numpy(), window_shape=self.dim, step=self.stride
        ))
        
        # encode the patches by vit
        frame = torch.zeros(
            (torch.tensor(patches.shape[:3]) * 16).int().tolist(),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        for z in range(patches.shape[0]):
            for y in range(patches.shape[1]):
                for x in range(patches.shape[2]):
                    patch = patches[z, y, x]
                    if torch.cuda.is_available(): patch = patch.to("cuda")
                    encode = self.vit(
                        patch.unsqueeze(0).unsqueeze(0)
                    ).squeeze(0).squeeze(0).reshape(16, 16, 16)
                    frame[
                        z*16:(z+1)*16, y*16:(y+1)*16, x*16:(x+1)*16
                    ] = encode

        # age
        age = torch.tensor(self.profile.loc[index]["age"]).float()

        return frame, age

    def __len__(self) -> int:
        return len(self.profile)


def getProfile(
    info_load_fold: str, tiff_load_fold: str, train_percentage: float = 0.6
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # sample list
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

    # get profile
    profile = {
        "sample": sample_list, "age": [], "sex": [], "shape": [],
        "info": [], "tiff": []
    }
    for sample in tqdm.tqdm(
        sample_list, smoothing=0.0, unit="sample", desc="getProfile"
    ):
        # path
        profile["info"].append(os.path.join(info_load_fold, sample + ".txt"))
        profile["tiff"].append(os.path.join(tiff_load_fold, sample + ".tif"))
        # read info
        with open(profile["info"][-1], 'r', encoding='utf-8') as file:
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
        with tifffile.TiffFile(profile["tiff"][-1]) as tif:
            profile["shape"].append((len(tif.pages), *tif.pages[0].shape))
    profile = pd.DataFrame(profile)

    # split profile by shape and sex
    profile_train = pd.DataFrame(columns=profile.columns)
    profile_valid = pd.DataFrame(columns=profile.columns)
    for shape in profile["shape"].unique():
        for sex in profile["sex"].unique():
            for age in range(7, 21):
                temp = profile
                temp = temp[temp["shape"] == shape]
                temp = temp[temp["sex"] == sex]
                temp = temp[temp["age"] >= age]
                temp = temp[temp["age"] < age + 1]
                temp_train = temp.sample(frac=train_percentage)
                temp_valid = temp.drop(temp_train.index)
                profile_train = temp_train if profile_train.empty else pd.concat(
                    [profile_train, temp_train]
                )
                profile_valid = temp_valid if profile_valid.empty else pd.concat(
                    [profile_valid, temp_valid]
                )

    return profile, profile_train, profile_valid


if __name__ == "__main__":
    import cfg

    _, profile_train, profile_valid = getProfile(
        cfg.info_load_fold, cfg.tiff_load_fold
    )

    if not os.path.exists(os.path.dirname(cfg.profile_train_path)):
        os.makedirs(os.path.dirname(cfg.profile_train_path))
    if not os.path.exists(os.path.dirname(cfg.profile_valid_path)):
        os.makedirs(os.path.dirname(cfg.profile_train_path))
    profile_train.to_csv(cfg.profile_train_path, index=False)
    profile_valid.to_csv(cfg.profile_valid_path, index=False)
