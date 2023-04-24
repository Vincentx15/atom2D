import os
import sys

from atom3d.datasets import LMDBDataset
import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.main import process_df  # noqa
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
        for name, dataframe in zip(names, structs_df):
            if name is None:
                continue
            else:
                if name in self.failed_set:
                    return 0, (name, index)
                try:
                    dump_surf_dir = self.get_geometry_dir(name)
                    dump_operator_dir = self.get_operator_dir(name)
                    is_valid_mesh = process_df(df=dataframe,
                                               name=name,
                                               dump_surf_dir=dump_surf_dir,
                                               dump_operator_dir=dump_operator_dir,
                                               recompute=False,
                                               verbose=False,
                                               clean_temp=False  # several chains are always computed,
                                               # making the computation buggy with cleaning
                                               )
                    if not is_valid_mesh:
                        self.failed_set.add(name)
                        self.print_error(name, index, "Invalid mesh")
                        return 0, (name, index)
                except Exception as e:
                    self.failed_set.add(name)
                    self.print_error(name, index, e)
                    return 0, (name, index)
        return 1, None


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
