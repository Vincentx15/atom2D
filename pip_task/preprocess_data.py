import os
import sys

from atom3d.datasets import LMDBDataset
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

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

    def get_geometry_dir(self, name):  # todo: if I don't define this function here, the class does not inherit it, why?
        return naming_utils.name_to_dir(name, dir_path=self.geometry_path)

    def get_operator_dir(self, name):  # todo: if I don't define this function here, the class does not inherit it, why?
        return naming_utils.name_to_dir(name, dir_path=self.operator_path)

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
                                               verbose=False)
                    if not is_valid_mesh:
                        self.failed_set.add(name)
                        print_error(name, index, "Invalid mesh")
                        return 0, (name, index)
                except Exception as e:
                    self.failed_set.add(name)
                    print_error(name, index, e)
                    return 0, (name, index)
        return 1, None


def print_error(name, index, error):
    print("--" * 20)
    print(f'Failed precomputing for {name}, index: {index}')
    print(error)
    print("--" * 20)


# Finally, we need to iterate to precompute all relevant surfaces and operators
def compute_operators_all(lmdb_path):
    dataset = PIPAtom3DDataset(lmdb_path=lmdb_path)
    success_codes = Parallel(n_jobs=-2)(delayed(lambda x, i: x[i])(dataset, i) for i in tqdm(range(len(dataset))))
    success_codes, failed_list = zip(*success_codes)
    failed_list = [x for x in failed_list if x is not None]

    print(f'{sum(success_codes)}/{len(success_codes)} processed')
    print(list(failed_list))

    # Save the failed set
    with open(os.path.join(lmdb_path, 'failed_set.txt'), 'w') as f:
        for name in failed_list:
            f.write(f'{name[0]}, {name[1]}' + '\n')


if __name__ == '__main__':
    pass

    np.random.seed(0)
    torch.manual_seed(0)

    compute_operators_all(lmdb_path='../data/PIP/DIPS-split/data/test/')

    # dataset = PIPAtom3DDataset(lmdb_path='../data/PIP/DIPS-split/data/test/')
    # dataset.run_preprocess()
    # A first run gave us 100k pdb in the DB.
    # 87300/87303 processed
    # Discoverd 108805 pdb
