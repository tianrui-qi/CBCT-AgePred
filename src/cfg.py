__all__ = ["UNetConfig", "PretrainConfig", "FinetuneConfig"]


info_load_fold = "data/info/"
tiff_load_fold = "data/tiff/"
profile_train_path = "data/profile_train.csv"
profile_valid_path = "data/profile_valid.csv"


class UNetConfig:
    def __init__(self) -> None:
        self.trainset = {   # src.data.UNetDataset
            # profile
            "profile_load_path": profile_train_path,
            # normalization
            "min_HU": 0,
            "max_HU": 4000,
            # augmentation
            "degrees": 5.0,
            "translation": 10,
            # mode
            "mode": "teeth",        # "whole" or "teeth"
        }
        self.validset = {   # src.data.UNetDataset
            # profile
            "profile_load_path": profile_valid_path,
            # normalization
            "min_HU": self.trainset["min_HU"],
            "max_HU": self.trainset["max_HU"],
            # augmentation
            "degrees": None,        # to disable augmentation
            "translation": None,    # to disable augmentation
            # mode
            "mode": self.trainset["mode"],
        }
        self.model = {      # src.model.UNetModel
            "lr": 5e-4,
            "unet_kwargs": {
                "feats": [1, 24, 48, 96, 192, 384],
                "num_classes": 1,
                "use_cbam": True,
                "use_res" : True,
            },
        }
        self.trainer = {    # src.runner.Trainer
            # train
            "max_epoch": -1,
            "accumu_steps": 16,
            # data
            "batch_size": 1,
            "num_workers": 4,
            # recoder
            "version": "unet1",     # None to use pl default version control
            "save_top_k": 10,
            "ckpt_load_path": "",
            "ckpt_load_lr": False,
        }


class PretrainConfig:
    def __init__(self) -> None:
        self.trainset = {   # src.data.PretrainDataset
            # dim
            "num_sample": 1000,
            "num_sampling": 100,
            "dim": [160, 160, 160],
            # profile
            "profile_load_path": profile_train_path,
            # normalization
            "min_HU": 0,
            "max_HU": 4000,
            # augmentation
            "degrees": 5.0,
        }
        self.validset = {   # src.data.PretrainDataset
            # dim
            "num_sample": 200,
            "num_sampling": 100,
            "dim": self.trainset["dim"],
            # profile
            "profile_load_path": profile_valid_path,
            # normalization
            "min_HU": self.trainset["min_HU"],
            "max_HU": self.trainset["max_HU"],
            # augmentation
            "degrees": None,
        }
        self.model = {      # src.model.PretrainModel
            "lr": 5e-4,
            "vit_kwargs": {
                "image_size": 160,          # D
                "frames": 160,              # H and W
                "image_patch_size": 16,     # D patch size
                "frame_patch_size": 16,     # H and W patch size
                "num_classes": 4096,        # 16 * 16 * 16
                "dim": 1024,
                "depth": 6,
                "heads": 8,
                "mlp_dim": 2048,
                "channels": 1,
                "dropout": 0.1,    
                "emb_dropout": 0.1,   
            },
            "mae_kwargs": {
                "decoder_dim"  : 512,       # paper showed good results with 512
                "masking_ratio": 0.75,      # paper recommended 75%
                "decoder_depth": 6,         # anywhere from 1 to 8
                "decoder_heads": 8,
            },
        }
        self.trainer = {    # src.runner.Trainer
            # train
            "max_epoch": -1,
            "accumu_steps": 100,
            # data
            "batch_size": 10,
            "num_workers": 5,
            # recoder
            "version": "pretrain1",     # None to use pl default version control
            "save_top_k": 10,
            "ckpt_load_path": "",
            "ckpt_load_lr": False,
        }


class FinetuneConfig:
    def __init__(self) -> None:
        self.trainset = {   # src.data.FinetuneDataset
            # dim
            "dim"   : [160, 160, 160], 
            "stride": [140, 140, 140],
            # profile
            "profile_load_path": profile_train_path,
            # vit
            "vit_kwargs": PretrainConfig().model["vit_kwargs"],
            "ckpt_load_path": "",
            # normalization
            "min_HU": 0,
            "max_HU": 4000,
        }
        self.validset = {   # src.data.FinetuneDataset
            # dim
            "dim"   : [160, 160, 160], 
            "stride": [140, 140, 140],
            # profile
            "profile_load_path": profile_valid_path,
            # vit
            "vit_kwargs": PretrainConfig().model["vit_kwargs"],
            "ckpt_load_path": "",
            # normalization
            "min_HU": self.trainset["min_HU"],
            "max_HU": self.trainset["max_HU"],
        }
        self.model = {      # src.model.FinetuneDataset
            "lr": 5e-4,
            "unet_kwargs": {
                "feats": [1, 32, 64, 128, 256],
                "num_classes": 1,
                "use_cbam": False,
                "use_res" : True,
            },
        }
        self.trainer = {    # src.runner.Trainer
            # train
            "max_epoch": -1,
            "accumu_steps": 1,
            # data
            "batch_size": 16,
            "num_workers": 2,
            # recoder
            "version": "finetune1",     # None to use pl default version control
            "save_top_k": 10,
            "ckpt_load_path": "",
            "ckpt_load_lr": False,
        }
