import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from pl_module import PIPModule
from data_processing.data_module import PLDataModule
from data_loader import PIPDataset


def main():
    seed = 2023
    pl.seed_everything(seed, workers=True)

    # init model
    model = PIPModule()

    # init logger
    tb_logger = TensorBoardLogger(save_dir="./logs")
    loggers = [tb_logger]

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()
    callbacks = [lr_logger]

    # init trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=100,
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
    datamodule = PLDataModule(PIPDataset, "../data/PIP/DIPS-split/data/")

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
