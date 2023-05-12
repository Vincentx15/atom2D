import os
import sys

import math
import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom2d_utils import atom3dutils
from data_processing import main
from data_processing.preprocessor_dataset import Atom3DDataset


class PIPDataset(Atom3DDataset):
    def __init__(self, lmdb_path, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=5,
                 geometry_path='../data/processed_data/geometry/',
                 operator_path='../data/processed_data/operator/',
                 recompute=False):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.recompute = recompute

    def _num_to_use(self, num_pos, num_neg):
        """
        Depending on the number of pos and neg of the system, we might want to use
            different amounts of positive or negative coordinates.

        :param num_pos:
        :param num_neg:
        :return:
        """

        if self.neg_to_pos_ratio == -1:
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
            if self.max_pos_regions_per_ensemble != -1:
                num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    @staticmethod
    def _get_res_pair_ca_coords(samples_df, structs_df):
        def _get_ca_coord(struct, res):
            coord = struct[(struct.residue == res) & (struct.name == 'CA')][['x', 'y', 'z']].values[0]
            return coord

        res_pairs = samples_df[['residue0', 'residue1']].values
        cas = []
        for (res0, res1) in res_pairs:
            try:
                coord0 = _get_ca_coord(structs_df[0], res0)
                coord1 = _get_ca_coord(structs_df[1], res1)
                cas.append((res0, res1, coord0, coord1))
            except Exception:
                pass
        return cas

    def __getitem__(self, index):
        """

        :param index:
        :return: pos and neg arrays of the 2 partners CA 3D coordinates shape N_{pos,neg}x 2x 3
                 and the geometry objects necessary to embed the surfaces
        """
        try:
            item = self._lmdb_dataset[index]

            # Subunits
            wrapped_names, wrapped_structs = atom3dutils.get_subunits(item['atoms_pairs'])

            bdf0, bdf1, udf0, udf1 = wrapped_structs
            name_bdf0, name_bdf1, name_udf0, name_udf1 = wrapped_names
            structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
            names_used = [name_udf0, name_udf1] if name_udf0 is not None else [name_bdf0, name_bdf1]

            # Get all positives and negative neighbors, filter out non-empty hetero/insertion_code
            pos_neighbors_df = item['atoms_neighbors']
            neg_neighbors_df = atom3dutils.get_negatives(pos_neighbors_df, structs_df[0], structs_df[1])
            non_heteros = []
            for df in structs_df:
                non_heteros.append(df[(df.hetero == ' ') & (df.insertion_code == ' ')].residue.unique())
            pos_neighbors_df = pos_neighbors_df[pos_neighbors_df.residue0.isin(non_heteros[0])
                                                & pos_neighbors_df.residue1.isin(non_heteros[1])]
            neg_neighbors_df = neg_neighbors_df[neg_neighbors_df.residue0.isin(non_heteros[0])
                                                & neg_neighbors_df.residue1.isin(non_heteros[1])]

            # Sample pos and neg samples
            num_pos = pos_neighbors_df.shape[0]
            num_neg = neg_neighbors_df.shape[0]
            num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)
            if pos_neighbors_df.shape[0] == num_pos_to_use:
                pos_samples_df = pos_neighbors_df.reset_index(drop=True)
            else:
                pos_samples_df = pos_neighbors_df.sample(num_pos_to_use, replace=True).reset_index(drop=True)
            if neg_neighbors_df.shape[0] == num_neg_to_use:
                neg_samples_df = neg_neighbors_df.reset_index(drop=True)
            else:
                neg_samples_df = neg_neighbors_df.sample(num_neg_to_use, replace=True).reset_index(drop=True)

            pos_pairs_cas = self._get_res_pair_ca_coords(pos_samples_df, structs_df)
            neg_pairs_cas = self._get_res_pair_ca_coords(neg_samples_df, structs_df)
            pos_pairs_cas_arrs = torch.from_numpy(np.asarray([[ca_data[2], ca_data[3]] for ca_data in pos_pairs_cas]))
            neg_pairs_cas_arrs = torch.from_numpy(np.asarray([[ca_data[2], ca_data[3]] for ca_data in neg_pairs_cas]))

            geom_feats_0 = main.get_diffnetfiles(name=names_used[0],
                                                 df=structs_df[0],
                                                 dump_surf_dir=self.get_geometry_dir(names_used[0]),
                                                 dump_operator_dir=self.get_operator_dir(names_used[0]),
                                                 recompute=self.recompute)
            geom_feats_1 = main.get_diffnetfiles(name=names_used[1],
                                                 df=structs_df[1],
                                                 dump_surf_dir=self.get_geometry_dir(names_used[1]),
                                                 dump_operator_dir=self.get_operator_dir(names_used[1]),
                                                 recompute=self.recompute)
            if geom_feats_0 is None or geom_feats_1 is None:
                return None, None, None, None, None, None
            return names_used[0], names_used[1], pos_pairs_cas_arrs, neg_pairs_cas_arrs, geom_feats_0, geom_feats_1
        except Exception as e:
            print("------------------")
            print(f"Error in __getitem__: {e}")
            return None, None, None, None, None, None


if __name__ == '__main__':
    data_dir = '../data/PIP/DIPS-split/data/test/'
    dataset = PIPDataset(data_dir)
    for i, data in enumerate(dataset):
        print(i)
        if i > 5:
            break
