from typing import List

class Config:
    def __init__(self):
        ## data
        # path
        self.info_fold: str = "D:/tao_ct/tao_ct_info"
        self.cbct_fold: str = "D:/tao_ct/tao_ct_desens"
        self.tiff_fold: str = "D:/tao_ct/tao_ct_tiff"
        # normalization
        self.min_HU: int = -1000
        self.max_HU: int =  5000
        # augmentation
        self.augmentation: bool = True
        self.degrees: float = 15.0
        self.translation: int = 20
        self.std: float = 0.01

        ## model
        self.feats: List[int] = [1, 64, 128, 256, 512, 1024]
        self.use_cbam: bool = True
        self.use_res : bool = True
