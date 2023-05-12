import os
import sys
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from pl_module import PIPModule
from data_processing.data_module import PLDataModule
from data_loader import PIPDataset


def main(cfg=None):
    command = f"{'_'.join(sys.argv[1:])}"
    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init model
    model = PIPModule(cfg)

    # init logger
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{command}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
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
    datamodule = PLDataModule(PIPDataset, cfg.dataset.data_dir, cfg.loader.batch_size_train)

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    main(cfg)
