from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Sampler


class SkipBatchesSampler(Sampler):
    def __init__(self, data_source, batches_to_skip=73200):  # 73200
        self.data_source = data_source
        self.batches_to_skip = batches_to_skip

        self.batch_size = 1

    def __iter__(self):
        num_samples = len(self.data_source)
        indices = list(range(num_samples))
        start_index = self.batches_to_skip * self.batch_size

        return iter(indices[start_index:])

    def __len__(self):
        num_samples = len(self.data_source)
        start_index = self.batches_to_skip * self.batch_size
        return num_samples - start_index


class PLDataModule(pl.LightningDataModule):
    def __init__(self, dataset, cfg):
        super().__init__()
        self.dataset = dataset
        self.data_dir = Path(cfg.dataset.data_dir)
        self.return_graph = cfg.model.use_graph or cfg.model.use_graph_only
        self.return_surface = not cfg.model.use_graph_only
        self.big_graphs = cfg.dataset.big_graphs
        self.cfg = cfg

    def train_dataloader(self):
        data_dir = self.data_dir / "train"
        dataset = self.dataset(data_dir, return_graph=self.return_graph, return_surface=self.return_surface,
                               big_graphs=self.big_graphs)
        # sampler = SkipBatchesSampler(dataset)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_train,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          shuffle=self.cfg.loader.shuffle, collate_fn=lambda x: x,
                          # sampler=sampler
                          )

    def val_dataloader(self):
        data_dir = self.data_dir / "val"
        dataset = self.dataset(data_dir, return_graph=self.return_graph, return_surface=self.return_surface,
                               big_graphs=self.big_graphs)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: x)

    def test_dataloader(self):
        data_dir = self.data_dir / "test"
        dataset = self.dataset(data_dir, return_graph=self.return_graph, return_surface=self.return_surface,
                               big_graphs=self.big_graphs)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: x)
