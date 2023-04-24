import os
import sys

from atom3d.datasets import LMDBDataset
import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom2d_utils import atom3dutils, naming_utils  # noqa
from data_processing.preprocessor_dataset import DryRunDataset, ProcessorDataset  # noqa


class PIPDryRunDataset(DryRunDataset):

    def __init__(self, lmdb_path):
        super().__init__(lmdb_path=lmdb_path)

    def __getitem__(self, index):
        """
        Return a list of subunit for this item.
        :param index:
        :return:
        """
        if self._lmdb_dataset is None:
            self._lmdb_dataset = LMDBDataset(self.lmdb_path)
        item = self._lmdb_dataset[index]

        names, _ = atom3dutils.get_subunits(item['atoms_pairs'])
        return [name for name in names if name is not None]


class PIPAtom3DDataset(ProcessorDataset):
    def __init__(self, lmdb_path,
                 geometry_path='../data/processed_data/geometry/',
                 operator_path='../data/processed_data/operator/',
                 subunits_mapping=None):
        super().__init__(lmdb_path=lmdb_path,
                         geometry_path=geometry_path,
                         operator_path=operator_path,
                         subunits_mapping=subunits_mapping)

    def __getitem__(self, index):
        if self._lmdb_dataset is None:
            self._lmdb_dataset = LMDBDataset(self.lmdb_path)
        unique_name, lmdb_id = self.systems_to_compute[index]
        lmdb_item = self._lmdb_dataset[lmdb_id]

        # Get subunits from this dataframe, bound and unbound forms of each complex
        # names : ('117e.pdb1.gz_1_A', '117e.pdb1.gz_1_B', None, None)
        names, dfs = atom3dutils.get_subunits(lmdb_item['atoms_pairs'])
        # TODO : make it better, without useless shit
        position = names.index(unique_name)
        return self.process_one(name=unique_name, df=dfs[position], index=index)

        # names, (bdf0, bdf1, udf0, udf1) = atom3dutils.get_subunits(lmdb_item['atoms_pairs'])
        # Use the unbound form when available to increase generalization
        # TODO:check how it's done in atom3D
        # Then turn it into the structures
        # structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
        # return self.process_lists(names=names, dfs=structs_df, index=index)


if __name__ == '__main__':
    pass

    np.random.seed(0)
    torch.manual_seed(0)

    for mode in ['test', 'train', 'validation']:
        print(f"Processing for PIP, {mode} set")
        data_dir = f'../data/PIP/DIPS-split/data/{mode}'
        subunits_mapping = PIPDryRunDataset(lmdb_path=data_dir).get_mapping()
        dataset = PIPAtom3DDataset(lmdb_path=data_dir, subunits_mapping=subunits_mapping)
        dataset.run_preprocess()
    # A first run gave us 100k pdb in the DB.
    # 87300/87303 processed
    # Discoverd 108805 pdb
