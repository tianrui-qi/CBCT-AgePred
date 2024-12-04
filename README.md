## Main

The main file `main.py` is use to train model. 
We have two model, one base on UNet, one base on ViT. 
The "unet" mode is for train UNet model, the "pretrain" mode is for pretrain
the ViT model, the "finetune" mode is for finetune the ViT model.

## Configuration

Config for module of different mode, store in `src/cfg.py`. 
The class name match the mode select
in `main.py`. For example, if select "unet", then the configuration 
`src/cfg.py/UNetConfig` will be use; if select "pretrain", then 
`src/cfg.py/PretrainConfig` will be use. Within each class, configuration for 
different module, i.e., train dataset, valid dataset, model, trainer, are group
by dictionary. All the configuration are enter in `main.py`.

## Data

Before preprocessing, run `util/profile.py` to generate a .csv file that include
metadata of all images. It will also split the dataset into train and valid set
and store as two different .csv file. Later we will use this .csv file to
retrieve the image. No raw image will be move or copy. 

Next, use `util/transfer.py` to preprocess the image. Note that this file may
modify/copy raw file. Make sure to read through the code before run it.

Then, `src/data.py` defined three kinds of dataset, "UNetDataset", 
"PretrainDataset", and "FinetuneDataset". Similar to configuration, the class
name match the mode select in `main.py`, i.e., if select "unet", then
"UNetDataset" will be use. Different from `util/transfer.py`, operation in
`src/data.py` is on the fly, no file will be move or copy.

## Model

Check `src/model/model.py` for three model we defined for UNet, Pretrain, and
Finetune. Similar to configuration and dataset, the class name match the mode
select in `main.py`.

## Trainer

We use pytorch lightning to train the model. The trainer is defined in
`src/trainer.py`.The trainer will use to save the model checkpoint, log the
training process, update the learning rate, etc.