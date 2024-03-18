__all__ = ["PretrainConfig", "FinetuneConfig"]


info_load_fold = "D:/info/"
tiff_load_fold = "D:/tiff/"
profile_train_path = "data/profile_train.csv"
profile_valid_path = "data/profile_valid.csv"


class PretrainConfig:
    def __init__(self) -> None:
        self.trainset = {   # src.data.PretrainDataset
            # dim
            "num_sample": 1000,
            "num_sampling": 100,
            "dim": [160, 160, 160],
            # profile
            "profile_load_path": profile_train_path,
            # augmentation
            "degrees": 5.0,
        }
        self.validset = {   # src.data.PretrainDataset
            # dim
            "num_sample": 200,
            "num_sampling": 100,
            "dim": [160, 160, 160],
            # profile
            "profile_load_path": profile_valid_path,
            # augmentation
            "degrees": None,
        }
        self.model = {
            "lr": 5e-4,
            "T_max": 15,                    # for cosine annealing
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
        self.trainer = {
            # train
            "max_epoch": -1,
            "accumu_steps": 100,
            # data
            "batch_size": 10,
            "num_workers": 5,
            # recoder
            "version": "pretrain",  # None to use pl default version control
            "save_top_k": 5,
            "ckpt_load_path": None,
        }


class FinetuneConfig:
    def __init__(self) -> None:
        self.trainset = {   # src.data.FinetuneDataset
            # dim
            "dim": [160, 160, 160], 
            "stride": [140, 140, 140],
            # profile
            "profile_load_path": profile_train_path,
        }
        self.validset = {   # src.data.FinetuneDataset
            # dim
            "dim": [160, 160, 160], 
            "stride": [140, 140, 140],
            # profile
            "profile_load_path": profile_valid_path,
        }
        self.model = {
            "lr": 5e-4,
            "T_max": 15,                    # for cosine annealing
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
        }
