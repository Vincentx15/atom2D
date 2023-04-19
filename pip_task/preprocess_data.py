import os
import sys

from atom3d.datasets import LMDBDataset
import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.main import process_df
from atom2d_utils import atom3dutils
from data_processing.PreprocessorDataset import ProcessorDataset


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

        # For pinpointing one pdb code that would be buggy
        # for name in names:
        #     if not "1jcc.pdb1.gz_1_C" in name:
        #         return
        # print('doing a buggy one')

        # Use the unbound form when available to increase generalization TODO:check how it's done in atom3D
        # Then turn it into the structures
        structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
        for name, dataframe in zip(names, structs_df):
            if name is None:
                continue
            else:
                if name in self.failed_set:
                    return 0
                try:
                    dump_surf_dir = self.get_geometry_dir(name)
                    dump_operator_dir = self.get_operator_dir(name)
                    process_df(df=dataframe,
                               name=name,
                               dump_surf_dir=dump_surf_dir,
                               dump_operator_dir=dump_operator_dir,
                               recompute=False)
                    # print(f'Precomputed successfully for {name}')
                except Exception:
                    self.failed_set.add(name)
                    # print("failed")
                    print(f'Failed precomputing for {name}')
                    return 0
        return 1


if __name__ == '__main__':
    pass

    np.random.seed(0)
    torch.manual_seed(0)

    dataset = PIPAtom3DDataset(lmdb_path='../data/PIP/DIPS-split/data/test/')
    dataset.run_preprocess()
    # A first run gave us 100k pdb in the DB.
    # 87300/87303 processed
    # Discoverd 108805 pdb
