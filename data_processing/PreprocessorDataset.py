import os
import sys

from atom3d.datasets import LMDBDataset
import numpy as np
import time
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom2d_utils import naming_utils


class ProcessorDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, geometry_path, operator_path):
        _lmdb_dataset = LMDBDataset(lmdb_path)
        self.length = len(_lmdb_dataset)
        self._lmdb_dataset = None
        self.failed_set = set()
        self.lmdb_path = lmdb_path
        self.geometry_path = geometry_path
        self.operator_path = operator_path

    def __len__(self) -> int:
        return self.length

    def run_preprocess(self):
        """
        Default class to go through a dataset without collating and batching, for preprocessing
        :param dataset:
        :return:
        """
        t0 = time.time()
        loader = torch.utils.data.DataLoader(self,
                                             # num_workers=0,
                                             num_workers=os.cpu_count(),
                                             batch_size=1,
                                             collate_fn=lambda x: x)
        for i, success in enumerate(loader):
            pass
            if not i % 100:
                print(f"Done {i} in {time.time() - t0}")

    def get_geometry_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.geometry_path)

    def get_operator_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.operator_path)

    def __getitem__(self, index):
        raise NotImplementedError


if __name__ == '__main__':
    pass

    np.random.seed(0)
    torch.manual_seed(0)

    dataset = ProcessorDataset(lmdb_path='example',
                               geometry_path='example',
                               operator_path='example_geo')
    dataset.run_preprocess()
