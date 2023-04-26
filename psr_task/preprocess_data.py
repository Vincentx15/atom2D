import os
import sys

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom3d.datasets import LMDBDataset
from data_processing.preprocessor_dataset import DryRunDataset, ProcessorDataset


class PSRDryRunDataset(DryRunDataset):

    def __init__(self, lmdb_path):
        super().__init__(lmdb_path=lmdb_path)

    def __getitem__(self, index):
        """
        Return a list of subunit for this item.
        :param index:
        :return:
        """
        item = self._lmdb_dataset[index]
        # item[id] has a weird formatting
        name = item['id']
        target, decoy = name[1:-1].split(',')
        target, decoy = target[2:-1], decoy[2:-1]
        name = f"{target}_{decoy}"
        return [name]


class PSRAtom3DDataset(ProcessorDataset):
    """

    """

    def __init__(self, lmdb_path,
                 subunits_mapping,
                 geometry_path='../data/PSR/geometry/',
                 operator_path='../data/PSR/operator/',
                 recompute=False,
                 verbose=False):
        super().__init__(lmdb_path=lmdb_path,
                         geometry_path=geometry_path,
                         operator_path=operator_path,
                         subunits_mapping=subunits_mapping,
                         recompute=recompute,
                         verbose=verbose)

    def __getitem__(self, index):
        _, lmdb_id = self.systems_to_compute[index]
        lmdb_item = self._lmdb_dataset[lmdb_id]

        df = lmdb_item['atoms'].reset_index(drop=True)
        # item[id] has a weird formatting
        name = lmdb_item['id']
        target, decoy = name[1:-1].split(',')
        target, decoy = target[2:-1], decoy[2:-1]
        name = f"{target}_{decoy}"
        return self.process_one(name=name, df=df, index=index)


if __name__ == '__main__':
    pass
    np.random.seed(0)
    torch.manual_seed(0)

    for mode in ['test', 'train', 'val']:
        print(f"Processing for PSR, {mode} set")
        data_dir = f'../data/PSR/{mode}'
        subunits_mapping = PSRDryRunDataset(lmdb_path=data_dir).get_mapping()
        dataset = PSRAtom3DDataset(lmdb_path=data_dir, subunits_mapping=subunits_mapping)
        dataset.run_preprocess()
