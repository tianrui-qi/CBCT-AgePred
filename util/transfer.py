import numpy as np
import pydicom
import tifffile
import os
import re
import json
import shutil

import tqdm
from typing import List


__all__ = []


def getNameList(
    profile_path: str, cbct_fold: str, filter: int = 10
) -> List[str]:
    """
    Go through the CBCT folder and get the shape of each patient's CBCT.
    Store key as shape and value as list of patient name with that shape in
    a dictionary. Save that dictionary as profile.json so that we do not need
    to go through the CBCT folder again next time. Then use the profile
    dictionary to filtering out patients with unexpected shapes, i.e., the
    number of patients with the same shape is less than input filter.

    Args:
        profile_path (str): The path to the profile JSON file.
        cbct_fold (str): The path to the CBCT folder.
        filter (int): The minimum number of patients with the same shape to
        not be filtered out. Defaults to 10.

    Returns:
        List[str]: A list of patient names same to that in CBCT folder after
        filtering.
    """

    profile = {}
    name_list = os.listdir(cbct_fold)

    # get the profile dictionary where key is shape of CBCT and value is list of 
    # corresponding patient name
    if os.path.exists(profile_path):
        # if profile.json exist, directly use the saved one to save time
        # load the profile json as dictionary
        with open(profile_path, 'r') as json_file:
            profile_convert = json.load(json_file)
        profile = {eval(key): value for key, value in profile_convert.items()}
    else:
        # if profile.json do not exist, read cbct of all patient and create new
        name_list = os.listdir(cbct_fold)
        for i in tqdm.tqdm(range(len(name_list)), desc="getNameList"):
            # path for current patient
            cbct_path = os.path.join(cbct_fold, name_list[i])
            # get D
            D = 0
            for file in os.listdir(cbct_path):
                if file.startswith("CT"): D += 1
            # get the H and W
            H, W = 0, 0
            for file in os.listdir(cbct_path):
                if file.startswith("CT"):
                    dcm_file = pydicom.dcmread(
                        os.path.join(cbct_path, file)
                    ).pixel_array
                    (H, W) = dcm_file.shape
                    break
            # save shape in dictionary
            if (D, H, W) not in profile: 
                profile[(D, H, W)] = []
            profile[(D, H, W)].append(name_list[i])
        # save the profile dictionary as json
        profile_convert = {str(key): value for key, value in profile.items()}
        with open(profile_path, 'w') as json_file:
            json.dump(profile_convert, json_file)

    # use profile dictionary to remove the patients with unexpected shape
    for key in profile:
        if len(profile[key]) < filter:
            for value in profile[key]:
                name_list.remove(value)

    # print infomation of profile and name list
    for key in profile: 
        if len(profile[key]) < filter: print(key, profile[key])
        else: print(key, len(profile[key]))
    print("number of sample after filter: {}".format(len(name_list)))

    return name_list


def transfer(
    name_list, 
    cbct_fold_src: str, info_fold_src: str, 
    tiff_fold_dst: str, info_fold_dst: str,
) -> None:
    """
    From the source directory, for each patient, we read CBCT images layer by 
    layer and save them as TIFF images in the destination directory. We also
    copy the info file of each patient from the source directory to the
    destination directory. Note that the copy of info file will be skipped if
    input info_fold_src and info_fold_dst are the same.

    Args:
        name_list (list): List of patient names after filtering.
        cbct_fold_src (str): Source directory containing CBCT images.
        info_fold_src (str): Source directory containing info files.
        tiff_fold_dst (str): Destination directory to save TIFF images.
        info_fold_dst (str): Destination directory to save info files.
    """
    if not os.path.exists(tiff_fold_dst): os.makedirs(tiff_fold_dst)
    if not os.path.exists(info_fold_dst): os.makedirs(info_fold_dst)

    for name in tqdm.tqdm(name_list, smoothing=0, unit="patients"):
        # info
        try:
            shutil.copy(
                os.path.join(info_fold_src, "{}.txt".format(name)), 
                os.path.join(info_fold_dst, "{}.txt".format(name))
            )
        except shutil.SameFileError:
            pass

        # get file list of cbct
        file_list = os.listdir(os.path.join(cbct_fold_src, name))
        def extract_number(filename):
            # extract the number before '.dcm' and after the last period
            match = re.search(r'\.(\d+)\.dcm$', filename)
            if match: return int(match.group(1))
            else: return None
        file_list = sorted(file_list, key=extract_number)

        # read cbct layer by layer
        frame = []
        for file in file_list:
            if not file.startswith("CT"): continue
            dcm_file = pydicom.dcmread(
                os.path.join(cbct_fold_src, name, file)
            )
            frame.append(dcm_file.pixel_array)
        frame = np.stack(frame)

        # save tiff
        tifffile.imwrite(
            os.path.join(tiff_fold_dst, "{}.tif".format(name)), frame
        )


if __name__ == "__main__":
    src = "D:/CBCT-AgePred/"
    dst = "D:/CBCT-AgePred/"
    part = 0
    # filtered name_list
    name_list = getNameList(
        profile_path=os.path.join(src, "profile.json"), 
        cbct_fold=os.path.join(src, "cbct/"),
        filter=10
    )
    # transfer part of name_list patient from src to dst
    transfer(
        name_list[1006*part:1006*(part+1)],
        cbct_fold_src=os.path.join(src, "cbct/"),
        info_fold_src=os.path.join(src, "info/"),
        tiff_fold_dst=os.path.join(dst, "tiff/"),
        info_fold_dst=os.path.join(dst, "info/"),
    )
