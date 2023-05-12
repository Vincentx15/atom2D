import os
import sys
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from pl_module import PSRModule
from data_processing.data_module import PLDataModule
from data_loader import PSRDataset


def main(cfg=None):
    seed = 2023
    pl.seed_everything(seed, workers=True)

    # init model
    model = PSRModule(cfg)
    # model = PSRModule.load_from_checkpoint(
    #     "/home/atom/github/atom2D/psr_task/logs/lightning_logs/version_1/checkpoints/epoch=99-step=2540000.ckpt")
    # a = sum(dict((p.data_ptr(), p.numel()) for p in model.model.parameters()).values())
    # print(a)
    # sys.exit()

    # init logger
    tb_logger = TensorBoardLogger(save_dir="./logs")
    loggers = [tb_logger]

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()
    callbacks = [lr_logger]

    # init trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[cfg.device],
        max_epochs=cfg.epochs,
        callbacks=callbacks,
        logger=loggers,
        # fast_dev_run=True,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=0.1,
        # profiler=True,
        # benchmark=True,
        # deterministic=True,
    )

    # datamodule
    datamodule = PLDataModule(PSRDataset, cfg.dataset.data_dir)


    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    main(cfg)
