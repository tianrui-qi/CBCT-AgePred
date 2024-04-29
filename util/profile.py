import pandas as pd

import os
import re
import tqdm
import tifffile


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
    info_load_fold = "data/info/"
    tiff_load_fold = "data/tiff/"
    profile_train_path = "data/profile_train.csv"
    profile_valid_path = "data/profile_valid.csv"

    _, profile_train, profile_valid = getProfile(
        info_load_fold, tiff_load_fold
    )

    if not os.path.exists(os.path.dirname(profile_train_path)):
        os.makedirs(os.path.dirname(profile_train_path))
    if not os.path.exists(os.path.dirname(profile_valid_path)):
        os.makedirs(os.path.dirname(profile_train_path))
    profile_train.to_csv(profile_train_path, index=False)
    profile_valid.to_csv(profile_valid_path, index=False)
