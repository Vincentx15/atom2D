import os
import sys
from pathlib import Path
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from pl_module import PIPModule
from data_processing.data_module import PLDataModule2
from data_loader import PIPDataset
from atom2d_utils.callbacks import CommandLoggerCallback


@hydra.main(config_path="./", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init model
    model = PIPModule(cfg)

    # init logger
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{auroc_val:.2f}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor="auroc_val",
        mode="max",
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        verbose=False,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor='auroc_val', patience=cfg.train.early_stoping_patience, mode='max')

    callbacks = [lr_logger, checkpoint_callback, early_stop_callback, CommandLoggerCallback(command)]

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
    datamodule = PLDataModule2(PIPDataset, cfg)

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
