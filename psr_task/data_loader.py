import os
import sys

import torch

from atom3d.datasets import LMDBDataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing import main
from data_processing.PreprocessorDataset import ProcessorDataset


class MSPDataset(ProcessorDataset):
    def __init__(self, lmdb_path,
                 geometry_path='../data/PSR/geometry/',
                 operator_path='../data/PSR/operator/'):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)

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

            geom_feats = main.get_diffnetfiles(name=name,
                                               df=df,
                                               dump_surf_dir=self.get_geometry_dir(name),
                                               dump_operator_dir=self.get_operator_dir(name))

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
