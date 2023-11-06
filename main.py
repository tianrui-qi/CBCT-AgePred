import torch.backends.cudnn

import config, data, model, runner

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    cfg = config.Config()
    # data
    patients = data.Patients(**cfg.data_config)
    # model
    resattnet = model.ResAttNet(**cfg.model_config)
    # runner
    trainer = runner.Trainer(
        dataset=patients, model=resattnet, **cfg.runner_config
    )

    # trainer.fit()
