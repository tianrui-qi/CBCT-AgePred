class Config:
    def __init__(self):
        self.data_config = {
            # path
            "info_fold": "/data/nanomega/projects/shanghai_tao/data/tao_ct/tao_ct_info",
            "cbct_fold": "/data/nanomega/projects/shanghai_tao/data/tao_ct/tao_ct_desens",
            "tiff_fold": "/data/nanomega/projects/shanghai_tao/data/tao_ct/tao_ct_tiff",
            # normalization
            "min_HU": -1000,
            "max_HU":  5000,
            # augmentation
            "degrees": 15.0,
            "translation": 20,
            "std": 0.01
        }
        self.model_config = {
            "feats": [1, 16, 32, 64, 128, 256, 512],
            "use_cbam": False,
            "use_res" : True
        }
        self.runner_config = {
            "max_epoch"      : 10000,
            "accumu_steps"   : 1,   # unit: batch
            "evalu_frequency": 10,  # unit: epoch
            # data
            "train_num" : 160,
            "batch_size": 8,
            "num_workers": 32,
            # path
            "ckpt_save_fold": "/data/nanomega/projects/shanghai_tao/ckpt",
            "ckpt_load_path": "",
            "ckpt_load_lr"  : False,
            # optimizer
            "lr"   : 1e-3,
            "gamma": 0.95
        }
