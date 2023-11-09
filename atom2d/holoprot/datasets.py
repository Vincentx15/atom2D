import os
import sys

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_sparse import SparseTensor

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.get_operators import surf_to_operators
from data_processing.data_module import AtomBatch
from data_processing.data_module import SurfaceObject


class HoloProtDataset(Dataset):
    def __init__(self, base_path, systems_ids, labels_all, labels_to_idx, use_wln=False):
        super().__init__()
        self.graph_path = os.path.join(base_path, 'backbone')
        self.surface_path = os.path.join(base_path, 'surface')
        self.operator_path = os.path.join(base_path, 'operators')
        self.systems_ids = systems_ids
        self.labels_all = labels_all
        self.labels_to_idx = labels_to_idx
        self.use_wln = use_wln

    def __len__(self):
        return len(self.systems_ids)

    def __getitem__(self, idx):
        """
        Return a list of subunit for this item.
        :param index:
        :return:
        """
        system = self.systems_ids[idx]
        y = self.labels_to_idx[self.labels_all[system]]
        system_as_path = system.upper() + '.pth'
        surf_path = os.path.join(self.surface_path, system_as_path)
        operator_path = os.path.join(self.operator_path, system_as_path.replace('.pth', '.npz'))
        graph_path = os.path.join(self.graph_path, system_as_path)
        try:
            graph = torch.load(graph_path)['prot']
            # Only keep distance features there if not using wln
            if not self.use_wln:
                graph.edge_attr = graph.edge_attr[:, 0]

            surface = torch.load(surf_path)['prot']
            frames, mass, _, evals, evecs, grad_x, grad_y = surf_to_operators(vertices=surface.vertices,
                                                                              faces=surface.faces,
                                                                              npz_path=operator_path)
            grad_x = SparseTensor.from_torch_sparse_coo_tensor(grad_x.float())
            grad_y = SparseTensor.from_torch_sparse_coo_tensor(grad_y.float())
            surface = SurfaceObject(features=surface.x, confidence=None, cat_confidence=False,
                                    vertices=surface.vertices, faces=surface.faces,
                                    L=torch.rand(1, 3), mass=mass, evals=evals, evecs=evecs, gradX=grad_x, gradY=grad_y)

            item = Data(surface=surface, graph=graph, y=torch.tensor([y]).long())
        except Exception as e:
            print(system, e)
            item = Data()
        return item


class HoloProtDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 raw_dir="../../data/holoprot/datasets/raw",
                 base_path="../../data/holoprot/datasets/processed/enzyme"):
        super().__init__()
        self.raw_dir = raw_dir
        self.base_path = base_path
        self.load_ids()
        self.cfg = cfg

    def load_ids(self):
        with open(f"{self.raw_dir}/metadata/base_split.json", "r") as f:
            splits = json.load(f)
        # splits = {
        #     'train': ['4fae_B', '102l_A'],
        #     'valid': ['4fae_B', '102l_A'],
        #     'test': ['4fae_B', '102l_A']
        # }
        self.splits = splits

        with open(f"{self.raw_dir}/metadata/function_labels.json", "r") as f:
            self.labels_all = json.load(f)

        with open(f"{self.raw_dir}/metadata/labels_to_idx.json", "r") as f:
            self.labels_to_idx = json.load(f)

        self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}

    def train_dataloader(self):
        dataset = HoloProtDataset(base_path=self.base_path, systems_ids=self.splits['train'],
                                  labels_all=self.labels_all, labels_to_idx=self.labels_to_idx,
                                  use_wln=self.cfg.model.use_wln)
        return DataLoader(dataset,
                          num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_train,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          shuffle=self.cfg.loader.shuffle, collate_fn=lambda x: AtomBatch.from_data_list(x)
                          )

    def val_dataloader(self):
        dataset = HoloProtDataset(base_path=self.base_path, systems_ids=self.splits['valid'],
                                  labels_all=self.labels_all, labels_to_idx=self.labels_to_idx,
                                  use_wln=self.cfg.model.use_wln)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: AtomBatch.from_data_list(x))

    def test_dataloader(self):
        dataset = HoloProtDataset(base_path=self.base_path, systems_ids=self.splits['test'],
                                  labels_all=self.labels_all, labels_to_idx=self.labels_to_idx,
                                  use_wln=self.cfg.model.use_wln)
        return DataLoader(dataset, num_workers=self.cfg.loader.num_workers, batch_size=self.cfg.loader.batch_size_val,
                          pin_memory=self.cfg.loader.pin_memory, prefetch_factor=self.cfg.loader.prefetch_factor,
                          collate_fn=lambda x: AtomBatch.from_data_list(x))


if __name__ == '__main__':
    pass
    from collections import namedtuple

    Cfg = namedtuple("cfg", "loader")
    LoaderCfg = namedtuple("loader", "batch_size_train batch_size_val num_workers prefetch_factor pin_memory shuffle")
    toy_cfg = Cfg(LoaderCfg(2, 2, 0, 2, False, False))
    data_module = HoloProtDataModule(cfg=toy_cfg)
    test_loader = data_module.test_dataloader()
    for i, x in enumerate(test_loader):
        if not i % 200:
            print("Done", i)
