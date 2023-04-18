import os
import sys

import torch
from torch.utils.data import Dataset

from atom3d.datasets import LMDBDataset
from atom3d.util.formats import get_coordinates_from_df

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing import main


class MSPDataset(Dataset):
    def __init__(self, lmdb_path,
                 geometry_path='../data/MSP/geometry/',
                 operator_path='../data/MSP/operator/'):
        _lmdb_dataset = LMDBDataset(lmdb_path)
        self.length = len(_lmdb_dataset)
        self._lmdb_dataset = None
        self.lmdb_path = lmdb_path

        self.geometry_path = geometry_path
        self.operator_path = operator_path

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _extract_mut_idx(df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        return torch.LongTensor(idx)

    def __getitem__(self, index):
        """

        :param index:
        :return: pos and neg arrays of the 2 partners CA 3D coordinates shape N_{pos,neg}x 2x 3
                 and the geometry objects necessary to embed the surfaces
        """

        try:
            if self._lmdb_dataset is None:
                self._lmdb_dataset = LMDBDataset(self.lmdb_path)
            item = self._lmdb_dataset[index]

            df = item['atoms'].reset_index(drop=True)
            # item[id] has a weird formatting
            name = item['id']
            target, decoy = name[1:-1].split(',')
            target, decoy = target[2:-1], decoy[2:-1]
            name = f"{target}_{decoy}"
            scores = item['scores']

            geom_feats = main.get_diffnetfiles(name=name, df=df,
                                               geometry_path=self.geometry_path,
                                               operator_path=self.operator_path)

            return name, geom_feats, scores
        except Exception as e:
            print("------------------")
            print(f"Error in __getitem__: {e}")
            return None, None, None, None, None, None


if __name__ == '__main__':
    data_dir = '../data/PSR/test/'
    dataset = MSPDataset(data_dir)
    for i, data in enumerate(dataset):
        print(i)
        if i > 5:
            break
