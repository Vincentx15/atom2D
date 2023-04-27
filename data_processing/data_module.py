from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class PLDataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_dir="path/to/data", loader_params=dict()):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.lparams = loader_params

    def train_dataloader(self):
        data_dir = self.data_dir / "train"
        dataset = self.dataset(data_dir)
        return DataLoader(dataset, num_workers=self.lparams.num_workers, batch_size=self.lparams.batch_size_train,
                          shuffle=True, collate_fn=lambda x: x)

    def val_dataloader(self):
        data_dir = self.data_dir / "val"
        dataset = self.dataset(data_dir)
        return DataLoader(dataset, num_workers=self.lparams.num_workers, batch_size=self.lparams.batch_size_train,
                          shuffle=False, collate_fn=lambda x: x)

    def test_dataloader(self):
        data_dir = self.data_dir / "test"
        dataset = self.dataset(data_dir)
        return DataLoader(dataset, num_workers=self.lparams.num_workers, batch_size=self.lparams.batch_size_train,
                          shuffle=False, collate_fn=lambda x: x)
