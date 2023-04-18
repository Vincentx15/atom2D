import os
import sys

from atom3d.datasets import LMDBDataset
import numpy as np
import time
import torch
from torch.utils.data import Dataset
import warnings

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.main import process_df
from atom2d_utils import atom3dutils, naming_utils

warnings.filterwarnings("ignore", message="In a future version of pandas, a length 1 tuple will be returned when")

"""
Here, we define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to efficiently do this preprocessing
"""


class MapAtom3DDataset(Dataset):
    def __init__(self, lmdb_path):
        _lmdb_dataset = LMDBDataset(lmdb_path)
        self.length = len(_lmdb_dataset)
        self._lmdb_dataset = None
        self.failed_set = set()
        self.lmdb_path = lmdb_path

    def __len__(self) -> int:
        return self.length

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
                    dump_surf_dir = os.path.join('../data/processed_data/geometry/',
                                                 naming_utils.name_to_dir(name))
                    dump_operator_dir = os.path.join('../data/processed_data/operator/',
                                                     naming_utils.name_to_dir(name))
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


# Finally, we need to iterate to precompute all relevant surfaces and operators
def compute_operators_all(data_dir):
    t0 = time.time()
    dataset = MapAtom3DDataset(data_dir)
    loader = torch.utils.data.DataLoader(dataset,
                                         # num_workers=0,
                                         num_workers=os.cpu_count(),
                                         batch_size=1,
                                         collate_fn=lambda x: x)
    for i, success in enumerate(loader):
        pass
        if not i % 100:
            print(f"Done {i} in {time.time() - t0}")


if __name__ == '__main__':
    pass

    np.random.seed(0)
    torch.manual_seed(0)

    compute_operators_all(data_dir='../data/PIP/DIPS-split/data/test/')

    # A first run gave us 100k pdb in the DB.
    # 87300/87303 processed
    # Discoverd 108805 pdb

