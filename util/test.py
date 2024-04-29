import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader, random_split
import torch.backends.cudnn

import numpy as np
import random
import os
import tifffile
import os
import re
import pandas as pd
import tqdm
from collections import OrderedDict

import src


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    """Set the seed for reproducibility in PyTorch for multi-GPU."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed

    # If you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensuring that all the GPUs have the same seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getProfile(info_load_fold, tiff_load_fold, profile_save_path):
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
    }
    for sample in tqdm.tqdm(
        sample_list, smoothing=0.0, unit="sample", desc="getProfile"
    ):
        # path
        info_load_path = os.path.join(info_load_fold, sample + ".txt")
        tiff_load_path = os.path.join(tiff_load_fold, sample + ".tif")
        # read info
        with open(info_load_path, 'r', encoding='utf-8') as file:
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
        with tifffile.TiffFile(tiff_load_path) as tif:
            profile["shape"].append((len(tif.pages), *tif.pages[0].shape))
    profile = pd.DataFrame(profile)

    # only keep sample that age in range 5 - 25
    profile = profile[(profile["age"] >= 5) | (profile["age"] <= 25)]
    # sort profile by age
    # profile = profile.sort_values(by="age", ascending=True)

    # save
    profile.to_csv(profile_save_path, index=False)

    return profile


class Patients(Dataset):
    def __init__(self, tiff_fold: str, info_fold: str, min_HU: int, max_HU: int,
    ) -> None:
        super(Patients, self).__init__()
        # path
        self.tiff_fold = tiff_fold
        self.info_fold = info_fold
        # normalization
        self.min_HU = min_HU
        self.max_HU = max_HU

        # cache
        self.name_list = [
            os.path.splitext(file)[0] for file in os.listdir(self.tiff_fold)
        ]

    def __getitem__(self, index: int):

        # read the tiff
        tiff = torch.from_numpy(tifffile.imread(
            os.path.join(self.tiff_fold, self.name_list[index] + ".tif")
        )).float()

        # normalization
        tiff = (tiff - self.min_HU) / (self.max_HU - self.min_HU)

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

        return self.name_list[index], torch.clip(tiff, 0, 1)

    def __len__(self) -> int:
        return len(self.name_list)


def getModel(ckpt_path):
    # initialize the model
    cfg = src.config.Config()
    resattnet = src.model.ResAttNet(**cfg.ResAttNet)
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs")
        resattnet = DataParallel(resattnet)
    resattnet.to("cuda")

    # load from checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    if isinstance(resattnet, DataParallel):
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = 'module.' + k if not k.startswith('module.') else k
            new_state_dict[name] = v
        checkpoint = new_state_dict
    else:
        checkpoint = checkpoint['model']

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    checkpoint = new_state_dict
    resattnet.load_state_dict(checkpoint)

    # other
    resattnet.half()
    resattnet.eval()

    return resattnet


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    min_HU =     0
    max_HU =  5000
    info_fold = "data/info"
    tiff_fold = "data/tiff"
    profile_path = "data/profile.csv"
    ckpt_path = "ckpt/04/76.ckpt"

    set_seed(42)
    # profile
    profile = getProfile(info_fold, tiff_fold, profile_path)
    # data
    dataset = Patients(
        tiff_fold=tiff_fold, info_fold=info_fold, min_HU=min_HU, max_HU=max_HU
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    # model
    resattnet = getModel(ckpt_path)
    # predict
    profile[ckpt_path] = 0.0
    with torch.no_grad():
        for i, (sample, frame) in tqdm.tqdm(
            enumerate(dataloader), total=len(dataset)
        ):
            predict = resattnet(frame.half().to("cuda")).float().cpu().detach().numpy()
            profile.loc[profile["sample"] == sample[0], ckpt_path] = predict
            profile.to_csv(profile_path, index=False)
