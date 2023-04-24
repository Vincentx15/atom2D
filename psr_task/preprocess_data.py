import os
import sys

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.main import process_df
from atom3d.datasets import LMDBDataset
from data_processing.preprocessor_dataset import ProcessorDataset

"""
Here, we define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to efficiently do this preprocessing
"""


class PSRAtom3DDataset(ProcessorDataset):
    """
    In this task, the loader returns two protein interfaces an original and mutated one.

    In the C3D and enn representation, a cropping of the protein around this interface is provided,
    whereas in the graph formulation, the whole graphs are used with an extra label indicating which nodes are mutated
    in the original and mutated versions.

    The model then does graph convolutions and add pool the node representations of the interfaces for each graph.
    Then, it feeds the concatenation of these representations to an MLP
    """

    def __init__(self, lmdb_path,
                 geometry_path='../data/PSR/geometry/',
                 operator_path='../data/PSR/operator/'):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)

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
            dump_surf_dir = self.get_geometry_dir(name)
            dump_operator_dir = self.get_operator_dir(name)
            process_df(df=df,
                       name=name,
                       dump_surf_dir=dump_surf_dir,
                       dump_operator_dir=dump_operator_dir,
                       recompute=False,
                       clean_temp=False)
            # print(f'Precomputed successfully for {name}')
        except Exception:
            print(f"failed for {name}")
            self.failed_set.add(name)
            # print(f'Failed precomputing for {name}')
            return 0
        return 1


if __name__ == '__main__':
    pass
    np.random.seed(0)
    torch.manual_seed(0)

    for mode in ['test', 'train', 'validation']:
        print(f"Processing for PSR, {mode} set")
        data_dir = f'../data/PSR/{mode}'
        dataset = PSRAtom3DDataset(lmdb_path=data_dir)
        dataset.run_preprocess()
