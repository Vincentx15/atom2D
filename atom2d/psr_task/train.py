import os
import sys

import hydra
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from pl_module import PSRModule
from data_processing.data_module import PLDataModule
from data_loader import PSRDataset
from atom2d_utils.callbacks import CommandLoggerCallback


@hydra.main(config_path="./", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init model
    model = PSRModule(cfg)

    # init logger
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{global_r_val:.2f}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor="global_r_val",
        mode="max",
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        verbose=False,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor='global_r_val', patience=cfg.train.early_stoping_patience,
                                                     mode='max')

    callbacks = [lr_logger, checkpoint_callback, early_stop_callback, CommandLoggerCallback(command)]

    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [cfg.device]}
    else:
        params = {}
    # init trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        val_check_interval=cfg.train.val_check_interval,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        limit_test_batches=cfg.train.limit_test_batches,
        overfit_batches=cfg.train.overfit_batches,
        # gradient clipping
        gradient_clip_val=cfg.train.gradient_clip_val,
        # fast_dev_run=True,
        # profiler=True,
        # benchmark=True,
        deterministic=cfg.train.deterministic,
        **params
    )

    # datamodule
    datamodule = PLDataModule(PSRDataset, cfg)

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
