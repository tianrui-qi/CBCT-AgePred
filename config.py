class Config:
    def __init__(self):
        self.ckpt_disk: str = "ckpt/"
        self.data_disk: str = "data/"
        self.Patients = {
            # path
            "cbct_fold": self.data_disk + "cbct/",
            "tiff_fold": self.data_disk + "tiff/",
            "info_fold": self.data_disk + "info/",
            # normalization
            "min_HU":     0,  # 02: -1000, 03: 0
            "max_HU":  5000,
            # augmentation
            "degrees": 10.0,
            "translation": 10,
            "std": 0.01
        }
        self.ResAttNet = {
            "feats": [1, 16, 32, 64, 128, 256, 512],
            "use_cbam": True,
            "use_res" : True
        }
        self.Trainer = {
            "max_epoch"      : 800,
            "accumu_steps"   : 1,   # unit: batch
            "evalu_frequency": 1,   # unit: epoch
            # path
            "ckpt_save_fold": self.ckpt_disk + "05",
            "ckpt_load_path": "",
            "ckpt_load_lr"  : False,
            # data
            "train_percent": 0.7,
            "batch_size" : 4,
            "num_workers": 16,
            # optimizer
            "lr"   : 1e-3,
        }
