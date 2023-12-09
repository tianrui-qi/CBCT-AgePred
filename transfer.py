import numpy as np
import pydicom
import tifffile
import os
import json
import shutil
import tqdm


def getNameList(profile_path: str, cbct_path: str):
    profile = {}
    name_list = os.listdir(cbct_path)

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
        name_list = os.listdir(cbct_path)
        for i in tqdm.tqdm(range(len(name_list)), desc="getNameList"):
            # path for current patient
            cbct_path_curr = os.path.join(cbct_path, name_list[i])
            # get D
            D = 0
            for file in os.listdir(cbct_path_curr):
                if file.startswith("CT"): D += 1
            # get the H and W
            H, W = 0, 0
            for file in os.listdir(cbct_path_curr):
                if file.startswith("CT"):
                    dcm_file = pydicom.dcmread(
                        os.path.join(cbct_path_curr, file)
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
        if len(profile[key]) < 2000:
            for value in profile[key]:
                name_list.remove(value)

    # print infomation of profile and name list
    """
    for key in profile: 
        if len(profile[key]) < 0: print(key, profile[key])
        else: print(key, len(profile[key]))
    print(len(name_list))
    """

    return name_list


if __name__ == "__main__":
    name_list = getNameList(
        profile_path="/nanolab/profile.json", cbct_path="/nanolab/cbct"
    )

    data_disk = "/data/nanomega/data/"
    tiff_fold = data_disk + "tiff/"
    info_fold = data_disk + "info/"
    if not os.path.exists(tiff_fold): os.makedirs(tiff_fold)
    if not os.path.exists(info_fold): os.makedirs(info_fold)

    for name in tqdm.tqdm(name_list[7:1006]):
        # info
        shutil.copy(
            "/nanolab/info/{}.txt".format(name), 
            info_fold + "{}.txt".format(name)
        )
        # read cbct layer by layer
        frame = []
        file_list = os.listdir("/nanolab/cbct/{}".format(name))
        for file in file_list:
            if not file.startswith("CT"): continue
            dcm_file = pydicom.dcmread(
                os.path.join("/nanolab/cbct/{}".format(name), file)
            )
            frame.append(dcm_file.pixel_array)
        frame = np.stack(frame)
        # save tiff
        tifffile.imwrite(tiff_fold + "{}.tif".format(name), frame)
