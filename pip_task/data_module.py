from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data_loader import PIPDataset
from preprocess_data import PIPDryRunDataset


class PIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="../data/PIP/DIPS-split/data/", batch_size=1):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

    def train_dataloader(self):
        data_dir = self.data_dir / "train"
        dataset = PIPDataset(data_dir)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=lambda x: x)

    def val_dataloader(self):
        data_dir = self.data_dir / "val"
        dataset = PIPDataset(data_dir)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=lambda x: x)

    def test_dataloader(self):
        data_dir = self.data_dir / "test"
        dataset = PIPDataset(data_dir)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=lambda x: x)
