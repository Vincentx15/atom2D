import os
import sys

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.preprocessor_dataset import DryRunDataset, ProcessorDataset


class MSPDryRunDataset(DryRunDataset):

    def __init__(self, lmdb_path):
        super().__init__(lmdb_path=lmdb_path)

    def __getitem__(self, index):
        """
        Return a list of subunit for this item.
        :param index:
        :return:
        """
        item = self._lmdb_dataset[index]

        # mutation is like AD56G which means Alanine (A) in chain D resnum 56 (D56) -> Glycine (G)
        pdb, chains_left, chains_right, mutation = item['id'].split('_')
        names = [f"{pdb}_{chains_left}", f"{pdb}_{chains_right}",
                 f"{pdb}_{chains_left}_{mutation}", f"{pdb}_{chains_right}_{mutation}"]

        return names


class MSPPreprocessDataset(ProcessorDataset):
    """
    In this task, the loader returns two protein interfaces an original and mutated one.

    In the C3D and enn representation, a cropping of the protein around this interface is provided,
    whereas in the graph formulation, the whole graphs are used with an extra label indicating which nodes are mutated
    in the original and mutated versions.

    The model then does graph convolutions and add pool the node representations of the interfaces for each graph.
    Then, it feeds the concatenation of these representations to an MLP
    """

    def __init__(self, lmdb_path,
                 subunits_mapping,
                 geometry_path='../data/MSP/geometry/',
                 operator_path='../data/MSP/operator/',
                 recompute=False,
                 verbose=False):
        super().__init__(lmdb_path=lmdb_path,
                         geometry_path=geometry_path,
                         operator_path=operator_path,
                         subunits_mapping=subunits_mapping,
                         recompute=recompute,
                         verbose=verbose)

    def __getitem__(self, index):
        unique_name, lmdb_id = self.systems_to_compute[index]
        lmdb_item = self._lmdb_dataset[lmdb_id]

        # mutation is like AD56G which means Alanine (A) in chain D resnum 56 (D56) -> Glycine (G)
        pdb, chains_left, chains_right, mutation = lmdb_item['id'].split('_')
        orig_df = lmdb_item['original_atoms'].reset_index(drop=True)
        mut_df = lmdb_item['mutated_atoms'].reset_index(drop=True)

        # Apparently this is faster than split
        left_orig = orig_df[orig_df['chain'].isin(list(chains_left))]
        right_orig = orig_df[orig_df['chain'].isin(list(chains_right))]
        left_mut = mut_df[mut_df['chain'].isin(list(chains_left))]
        right_mut = mut_df[mut_df['chain'].isin(list(chains_right))]

        names = [f"{pdb}_{chains_left}", f"{pdb}_{chains_right}",
                 f"{pdb}_{chains_left}_{mutation}", f"{pdb}_{chains_right}_{mutation}"]
        dfs = [left_orig, right_orig, left_mut, right_mut]

        # TODO : make it better, without useless shit
        position = names.index(unique_name)
        return self.process_one(name=unique_name, df=dfs[position], index=index)


if __name__ == '__main__':
    pass
    np.random.seed(0)
    torch.manual_seed(0)

    for mode in ['test', 'train', 'val']:
        print(f"Processing for MSP, {mode} set")
        data_dir = f'../data/MSP/{mode}'
        subunits_mapping = MSPDryRunDataset(lmdb_path=data_dir).get_mapping()
        dataset = MSPPreprocessDataset(lmdb_path=data_dir, subunits_mapping=subunits_mapping)
        dataset.run_preprocess()
