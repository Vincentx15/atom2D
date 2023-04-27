from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class PLDataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_dir="path/to/data", batch_size=1):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.dataset = dataset

    def train_dataloader(self):
        data_dir = self.data_dir / "train"
        dataset = self.dataset(data_dir)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=lambda x: x)

    def val_dataloader(self):
        data_dir = self.data_dir / "val"
        dataset = self.dataset(data_dir)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=lambda x: x)

    def test_dataloader(self):
        data_dir = self.data_dir / "test"
        dataset = self.dataset(data_dir)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=lambda x: x)