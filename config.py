class Config:
    def __init__(self):
        self.data_config = {
            # path
            "info_fold": "D:/tao_ct/tao_ct_info",
            "cbct_fold": "D:/tao_ct/tao_ct_desens",
            "tiff_fold": "D:/tao_ct/tao_ct_tiff",
            # normalization
            "min_HU": -1000,
            "max_HU":  5000,
            # augmentation
            "degrees": 15.0,
            "translation": 20,
            "std": 0.01
        }
        self.model_config = {
            "feats": [1, 64, 128, 256, 512, 1024],
            "use_cbam": True,
            "use_res" : True
        }
        self.runner_config = {
            "max_epoch"      : 1000,
            "accumu_steps"   : 1,
            "evalu_frequency": 10,
            # data
            "train_pct" : 0.6,
            "batch_size": 1,
            # path
            "ckpt_save_fold": "ckpt",
            "ckpt_load_path": "",
            "ckpt_load_lr"  : False,
            # optimizer
            "lr"   : 1e-3,
            "gamma": 0.95
        }
