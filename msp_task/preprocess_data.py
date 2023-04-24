import os
import sys

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.preprocessor_dataset import ProcessorDataset
from atom3d.datasets import LMDBDataset


class MSPAtom3DDataset(ProcessorDataset):
    """
    In this task, the loader returns two protein interfaces an original and mutated one.

    In the C3D and enn representation, a cropping of the protein around this interface is provided,
    whereas in the graph formulation, the whole graphs are used with an extra label indicating which nodes are mutated
    in the original and mutated versions.

    The model then does graph convolutions and add pool the node representations of the interfaces for each graph.
    Then, it feeds the concatenation of these representations to an MLP
    """

    def __init__(self, lmdb_path, geometry_path='../data/MSP/geometry/', operator_path='../data/MSP/operator/'):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)

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
        return self.process_lists(names=names, dfs=dfs, index=index)


if __name__ == '__main__':
    pass
    np.random.seed(0)
    torch.manual_seed(0)

    for mode in ['test', 'train', 'val']:
        print(f"Processing for MSP, {mode} set")
        data_dir = f'../data/MSP/{mode}'
        dataset = MSPAtom3DDataset(lmdb_path=data_dir)
        dataset.run_preprocess()
