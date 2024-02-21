class Config:
    def __init__(self):
        self.ckpt_disk: str = "ckpt/"
        self.data_disk: str = "data/"
        self.Patients = {
            # path
            "tiff_fold": self.data_disk + "tiff/",
            "info_fold": self.data_disk + "info/",
            # normalization
            "min_HU": -1000,
            "max_HU":  5000,
            # augmentation
            "degrees": 10.0,
            "translation": 10,
            "std": 0.01
        }
        self.ViT = {
            "image_size": 416,          # image size
            "frames": 416,              # number of frames
            "image_patch_size": 26,     # image patch size
            "frame_patch_size": 26,     # frame patch size
            "num_classes": 1000,
            "dim": 1024,
            "depth": 6,
            "heads": 8,
            "mlp_dim": 2048,
            "channels": 1,
            "dropout": 0.1,    
            "emb_dropout": 0.1,
        }
        self.MAE = {
            "decoder_dim": 512,     # paper showed good results with just 512
            "masking_ratio": 0.75,  # the paper recommended 75% masked patches
            "decoder_depth": 6,     # anywhere from 1 to 8
            "decoder_heads": 8,
        }
        self.Trainer = {
            "max_epoch"      : 400,
            "accumu_steps"   : 1,   # unit: batch
            "evalu_frequency": 1,   # unit: epoch
            # path
            "ckpt_save_fold": self.ckpt_disk + "01",
            "ckpt_load_path": "",
            "ckpt_load_lr"  : False,
            # data
            "train_percent": 0.6,
            "batch_size" : 1,
            "num_workers": 4,
            # optimizer
            "lr"   : 5e-4,
        }
