import os
import sys

import numpy as np
import os
import time
import torch
from torch.utils.data import Dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.main import process_df
from atom2d_utils import naming_utils
from atom3d.datasets import LMDBDataset

"""
Here, we define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to efficiently do this preprocessing
"""


class PSRAtom3DDataset(Dataset):
    """
    In this task, the loader returns two protein interfaces an original and mutated one.

    In the C3D and enn representation, a cropping of the protein around this interface is provided,
    whereas in the graph formulation, the whole graphs are used with an extra label indicating which nodes are mutated
    in the original and mutated versions.

    The model then does graph convolutions and add pool the node representations of the interfaces for each graph.
    Then, it feeds the concatenation of these representations to an MLP
    """

    def __init__(self, lmdb_path):
        _lmdb_dataset = LMDBDataset(lmdb_path)
        self.length = len(_lmdb_dataset)
        self._lmdb_dataset = None
        self.failed_set = set()
        self.lmdb_path = lmdb_path

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if self._lmdb_dataset is None:
            self._lmdb_dataset = LMDBDataset(self.lmdb_path)
        item = self._lmdb_dataset[index]

        df = item['atoms'].reset_index(drop=True)
        # item[id] has a weird formatting
        name = item['id']
        target, decoy = name[1:-1].split(',')
        target, decoy = target[2:-1], decoy[2:-1]
        name = f"{target}_{decoy}"

        try:
            dump_surf_dir = os.path.join('../data/PSR/geometry/', naming_utils.name_to_dir(name))
            dump_operator_dir = os.path.join('../data/PSR/operator/', naming_utils.name_to_dir(name))
            process_df(df=df,
                       name=name,
                       dump_surf_dir=dump_surf_dir,
                       dump_operator_dir=dump_operator_dir,
                       recompute=False)
            # print(f'Precomputed successfully for {name}')
        except Exception:
            print(f"failed for {name}")
            self.failed_set.add(name)
            # print(f'Failed precomputing for {name}')
            return 0
        return 1


# Finally, we need to iterate to precompute all relevant surfaces and operators
def compute_operators_all(data_dir):
    t0 = time.time()
    dataset = PSRAtom3DDataset(data_dir)
    loader = torch.utils.data.DataLoader(dataset,
                                         num_workers=0,
                                         # num_workers=os.cpu_count(),
                                         batch_size=1,
                                         collate_fn=lambda x: x)
    for i, success in enumerate(loader):
        pass
        if not i % 100:
            print(f"Done {i} in {time.time() - t0}")


if __name__ == '__main__':
    pass

    data_dir = '../data/PSR/test'

    np.random.seed(0)
    torch.manual_seed(0)

    compute_operators_all(data_dir=data_dir)
