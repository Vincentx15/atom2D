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


class MSPAtom3DDataset(Dataset):
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

        # mutation is like AD56G which means Alanine (A) in chain D resnum 56 (D56) -> Glycine (G)
        pdb, chains_left, chains_right, mutation = item['id'].split('_')
        orig_df = item['original_atoms'].reset_index(drop=True)
        mut_df = item['mutated_atoms'].reset_index(drop=True)

        # Apparently this is faster than split
        left_orig = orig_df[orig_df['chain'].isin(list(chains_left))]
        right_orig = orig_df[orig_df['chain'].isin(list(chains_right))]
        left_mut = mut_df[mut_df['chain'].isin(list(chains_left))]
        right_mut = mut_df[mut_df['chain'].isin(list(chains_right))]

        names = [f"{pdb}_{chains_left}", f"{pdb}_{chains_right}",
                 f"{pdb}_{chains_left}_{mutation}", f"{pdb}_{chains_right}_{mutation}"]
        dfs = [left_orig, right_orig, left_mut, right_mut]
        try:
            for name, df in zip(names, dfs):
                dump_surf_dir = os.path.join('../data/MSP/geometry/', naming_utils.name_to_dir(name))
                dump_operator_dir = os.path.join('../data/MSP/operator/', naming_utils.name_to_dir(name))
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
    dataset = MSPAtom3DDataset(data_dir)
    loader = torch.utils.data.DataLoader(dataset,
                                         # num_workers=0,
                                         num_workers=os.cpu_count(),
                                         batch_size=1,
                                         collate_fn=lambda x: x)
    for i, success in enumerate(loader):
        pass
        if not i % 100:
            print(f"Done {i} in {time.time() - t0}")


if __name__ == '__main__':
    pass

    data_dir = '../data/MSP/test'

    np.random.seed(0)
    torch.manual_seed(0)

    compute_operators_all(data_dir=data_dir)
