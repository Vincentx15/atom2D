import os
import sys

from atom3d.datasets import LMDBDataset
import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom2d_utils import atom3dutils, naming_utils  # noqa
from data_processing.preprocessor_dataset import ProcessorDataset  # noqa


class PIPAtom3DDataset(ProcessorDataset):
    def __init__(self, lmdb_path,
                 geometry_path='../data/processed_data/geometry/',
                 operator_path='../data/processed_data/operator/'):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)

    def __getitem__(self, index):
        if self._lmdb_dataset is None:
            self._lmdb_dataset = LMDBDataset(self.lmdb_path)
        item = self._lmdb_dataset[index]

        # Get subunits from this dataframe, bound and unbound forms of each complex
        # names : ('117e.pdb1.gz_1_A', '117e.pdb1.gz_1_B', None, None)
        names, (bdf0, bdf1, udf0, udf1) = atom3dutils.get_subunits(item['atoms_pairs'])

        # Use the unbound form when available to increase generalization
        # TODO:check how it's done in atom3D
        # Then turn it into the structures
        structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
        return self.process_lists(names=names, dfs=structs_df, index=index)


if __name__ == '__main__':
    pass

    np.random.seed(0)
    torch.manual_seed(0)

    for mode in ['test', 'train', 'validation']:
        print(f"Processing for PIP, {mode} set")
        data_dir = f'../data/PIP/DIPS-split/data/{mode}'
        dataset = PIPAtom3DDataset(lmdb_path=data_dir)
        dataset.run_preprocess()
    # A first run gave us 100k pdb in the DB.
    # 87300/87303 processed
    # Discoverd 108805 pdb
