__all__ = ["PretrainConfig", "FinetuneConfig"]


class PretrainConfig:
    def __init__(self) -> None:
        self.trainset = {   # src.data.PretrainDataset
            # dim
            "num": 10000, 
            "num_sampling": 100,
            "dim": [160, 160, 160],
            # path
            "tiff_load_fold": "D:/tiff/",   # TODO: split
            # augmentation
            "degrees": 5.0,
        }
        self.validset = {   # src.data.PretrainDataset
            # dim
            "num":  2000, 
            "num_sampling": 100,
            "dim": [160, 160, 160],
            # path
            "tiff_load_fold": "D:/tiff/",   # TODO: split
            # augmentation
            "degrees": None,
        }


class FinetuneConfig:
    def __init__(self) -> None:
        self.trainset = {   # src.data.FinetuneDataset
            # dim
            "dim": [160, 160, 160], 
            "stride": [140, 140, 140],
            # path
            "info_load_fold": "D:/info/",   # TODO: split
            "tiff_load_fold": "D:/tiff/",   # TODO: split
        }
        self.validset = {   # src.data.FinetuneDataset
            # dim
            "dim": [160, 160, 160], 
            "stride": [140, 140, 140],
            # path
            "info_load_fold": "D:/info/",   # TODO: split
            "tiff_load_fold": "D:/tiff/",   # TODO: split
        }


class Config:
    def __init__(self):
        self.MInterface = {
            "stage": "pretrain",            # pretrain or finetune
            "lr": 5e-4,
            "T_max": 10,                    # for cosine annealing
            "vit_kwargs": {
                "image_size": 160,          # image size
                "frames": 160,              # number of frames
                "image_patch_size": 20,     # image patch size
                "frame_patch_size": 20,     # frame patch size
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
                "decoder_dim": 512,         # paper showed good results with 512
                "masking_ratio": 0.75,      # paper recommended 75%
                "decoder_depth": 6,         # anywhere from 1 to 8
                "decoder_heads": 8,
            },
            "fc_kwargs": {
                "feats": [4096, 4096, 2048, 2048, 1]
            },
        }
        self.Trainer = {
            "accumulate_grad_batches": 128,
        }
