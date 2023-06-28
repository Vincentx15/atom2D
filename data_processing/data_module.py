from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class PLDataModule(pl.LightningDataModule):
    def __init__(self, dataset, cfg):
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(cfg.dataset.data_dir)
        self.return_graph = cfg.model.use_graph or cfg.model.use_graph_only
        self.cfg = cfg

    def train_dataloader(self):
        data_dir = self.data_dir / "train"
        dataset = self.dataset(data_dir, return_graph=self.return_graph)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_train,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: x)

    def val_dataloader(self):
        data_dir = self.data_dir / "val"
        dataset = self.dataset(data_dir, return_graph=self.return_graph)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: x)

    def test_dataloader(self):
        data_dir = self.data_dir / "test"
        dataset = self.dataset(data_dir, return_graph=self.return_graph)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: x)
