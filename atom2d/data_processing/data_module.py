from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor
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
                          shuffle=self.cfg.loader.shuffle, collate_fn=lambda x: AtomBatch.from_data_list(x),
                          # sampler=sampler
                          )

    def val_dataloader(self):
        data_dir = self.data_dir / "val"
        dataset = self.dataset(data_dir, return_graph=self.return_graph, return_surface=self.return_surface,
                               big_graphs=self.big_graphs)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: AtomBatch.from_data_list(x))

    def test_dataloader(self):
        data_dir = self.data_dir / "test"
        dataset = self.dataset(data_dir, return_graph=self.return_graph, return_surface=self.return_surface,
                               big_graphs=self.big_graphs)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: AtomBatch.from_data_list(x))


class AtomBatch(Data):
    def __init__(self, batch=None, **kwargs):
        super().__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data

    @staticmethod
    def from_data_list(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = AtomBatch()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            elif bool(re.search('(locs_left|locs_right|neg_stack|pos_stack)', key)):
                batch[key] = batch[key]
            elif key == 'labels_pip':
                batch[key] = torch.cat(batch[key])
            elif torch.is_tensor(item):
                batch[key] = torch.stack(batch[key])
            elif isinstance(item, SurfaceObject):
                batch[key] = SurfaceObject.from_data_list(batch[key])
            elif isinstance(item, Data):
                batch[key] = Batch.from_data_list([x for x in batch[key] if x is not None])
                batch[key] = batch[key] if batch[key].num_graphs > 0 else None
            elif isinstance(item, list):
                batch[key] = batch[key]
            elif isinstance(item, str):
                batch[key] = batch[key]
            elif isinstance(item, SparseTensor):
                batch[key] = batch[key]
            else:
                raise ValueError(f"Unsupported attribute type: {type(item)}, item : {item}, key : {key}")

        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class SurfaceObject(Data):
    def __init__(self, features=None, confidence=None, vertices=None,
                 mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None, faces=None, cat_confidence=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.vertices = vertices
        self.faces = faces
        if cat_confidence and confidence is not None:
            self.x = torch.cat((features, confidence[..., None]), dim=-1)
        else:
            self.x = features
        self.mass = mass
        self.L = L
        self.evals = evals
        self.evecs = evecs
        self.gradX = gradX
        self.gradY = gradY

    @classmethod
    def from_data_list(cls, data_list):
        # filter out None
        data_list = [data for data in data_list if data is not None]
        if len(data_list) == 0:
            return None

        # create batch
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = cls()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            elif torch.is_tensor(item):
                batch[key] = batch[key]
            elif isinstance(item, SparseTensor):
                batch[key] = batch[key]

        return batch.contiguous()
