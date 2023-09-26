import os
import sys

import hydra
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from pl_module import PIPModule
from data_processing.data_module import PLDataModule
from data_loader import NewPIP
from atom2d_utils.callbacks import CommandLoggerCallback


@hydra.main(config_path="./", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init model
    model = PIPModule(cfg)
    # load saved model
    saved_model_path = Path(__file__).resolve().parent / "lightning_logs" / cfg.path_model
    model.load_state_dict(torch.load(saved_model_path, map_location="cpu")["state_dict"])

    # init logger
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    # callbacks

    callbacks = [CommandLoggerCallback(command)]

    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [cfg.device]}
    else:
        params = {}

    # init trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        callbacks=callbacks,
        logger=loggers,
        limit_test_batches=cfg.train.limit_test_batches,
        # gradient clipping
        gradient_clip_val=cfg.train.gradient_clip_val,
        # fast_dev_run=True,
        # profiler=True,
        # benchmark=True,
        # deterministic=True,
        **params
    )

    # datamodule
    datamodule = PLDataModule(NewPIP, cfg)

    # test
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
