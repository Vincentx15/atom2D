import os
import sys

import math
import time

import numpy as np
import pandas as pd
import torch
import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom2d_utils import naming_utils
from atom2d_utils import atom3dutils
from data_processing import main
from data_processing.preprocessor_dataset import Atom3DDataset


class PIPReprocess(Atom3DDataset):
    def __init__(self, lmdb_path,
                 geometry_path='../data/processed_data/geometry/',
                 operator_path='../data/processed_data/operator/',
                 recompute=False):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)
        self.neg_to_pos_ratio = 1
        self.max_pos_regions_per_ensemble = 5
        self.recompute = recompute
        self.dump_dir = os.path.join(os.path.dirname(lmdb_path), 'systems')

    # def _num_to_use(self, num_pos, num_neg):
    #     """
    #     Depending on the number of pos and neg of the system, we might want to use
    #         different amounts of positive or negative coordinates.
    #
    #     :param num_pos:
    #     :param num_neg:
    #     :return:
    #     """
    #
    #     if self.neg_to_pos_ratio == -1:
    #         num_pos_to_use, num_neg_to_use = num_pos, num_neg
    #     else:
    #         num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
    #         if self.max_pos_regions_per_ensemble != -1:
    #             num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
    #         num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
    #     num_pos_to_use = int(math.ceil(num_pos_to_use))
    #     num_neg_to_use = int(math.ceil(num_neg_to_use))
    #     return num_pos_to_use, num_neg_to_use
    # @staticmethod
    # def _get_res_pair_ca_coords(samples_df, structs_df):
    #     def _get_ca_coord(struct, res):
    #         coord = struct[(struct.residue == res) & (struct.name == 'CA')][['x', 'y', 'z']].values[0]
    #         return coord
    #
    #     res_pairs = samples_df[['residue0', 'residue1']].values
    #     cas = []
    #     for (res0, res1) in res_pairs:
    #         try:
    #             coord0 = _get_ca_coord(structs_df[0], res0)
    #             coord1 = _get_ca_coord(structs_df[1], res1)
    #             cas.append((res0, res1, coord0, coord1))
    #         except Exception:
    #             pass
    #     return cas
    # def original_getitem(self, index):
    #     item = self._lmdb_dataset[index]
    #
    #     # Subunits
    #     wrapped_names, wrapped_structs = atom3dutils.get_subunits(item['atoms_pairs'])
    #
    #     bdf0, bdf1, udf0, udf1 = wrapped_structs
    #     name_bdf0, name_bdf1, name_udf0, name_udf1 = wrapped_names
    #     structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
    #     names_used = [name_udf0, name_udf1] if name_udf0 is not None else [name_bdf0, name_bdf1]
    #
    #     # Get all positives and negative neighbors, filter out non-empty hetero/insertion_code
    #     pos_neighbors_df = item['atoms_neighbors']
    #     neg_neighbors_df = atom3dutils.get_negatives(pos_neighbors_df, structs_df[0], structs_df[1])
    #     non_heteros = []
    #     for df in structs_df:
    #         non_heteros.append(df[(df.hetero == ' ') & (df.insertion_code == ' ')].residue.unique())
    #     pos_neighbors_df = pos_neighbors_df[pos_neighbors_df.residue0.isin(non_heteros[0])
    #                                         & pos_neighbors_df.residue1.isin(non_heteros[1])]
    #     neg_neighbors_df = neg_neighbors_df[neg_neighbors_df.residue0.isin(non_heteros[0])
    #                                         & neg_neighbors_df.residue1.isin(non_heteros[1])]
    #
    #     # Sample pos and neg samples
    #     num_pos = pos_neighbors_df.shape[0]
    #     num_neg = neg_neighbors_df.shape[0]
    #     num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)
    #     if pos_neighbors_df.shape[0] == num_pos_to_use:
    #         pos_samples_df = pos_neighbors_df.reset_index(drop=True)
    #     else:
    #         pos_samples_df = pos_neighbors_df.sample(num_pos_to_use, replace=True).reset_index(drop=True)
    #     if neg_neighbors_df.shape[0] == num_neg_to_use:
    #         neg_samples_df = neg_neighbors_df.reset_index(drop=True)
    #     else:
    #         neg_samples_df = neg_neighbors_df.sample(num_neg_to_use, replace=True).reset_index(drop=True)
    #
    #     pos_pairs_cas = self._get_res_pair_ca_coords(pos_samples_df, structs_df)
    #     neg_pairs_cas = self._get_res_pair_ca_coords(neg_samples_df, structs_df)
    #     pos_pairs_cas_arrs = torch.from_numpy(np.asarray([[ca_data[2], ca_data[3]] for ca_data in pos_pairs_cas]))
    #     neg_pairs_cas_arrs = torch.from_numpy(np.asarray([[ca_data[2], ca_data[3]] for ca_data in neg_pairs_cas]))
    #
    #     geom_feats_0 = main.get_diffnetfiles(name=names_used[0],
    #                                          df=structs_df[0],
    #                                          dump_surf_dir=self.get_geometry_dir(names_used[0]),
    #                                          dump_operator_dir=self.get_operator_dir(names_used[0]),
    #                                          recompute=self.recompute)
    #     geom_feats_1 = main.get_diffnetfiles(name=names_used[1],
    #                                          df=structs_df[1],
    #                                          dump_surf_dir=self.get_geometry_dir(names_used[1]),
    #                                          dump_operator_dir=self.get_operator_dir(names_used[1]),
    #                                          recompute=self.recompute)
    #     return names_used[0], names_used[1], pos_pairs_cas_arrs, neg_pairs_cas_arrs, geom_feats_0, geom_feats_1

    def reprocess(self, index):
        item = self._lmdb_dataset[index]

        # Subunits
        wrapped_names, wrapped_structs = atom3dutils.get_subunits(item['atoms_pairs'])

        bdf0, bdf1, udf0, udf1 = wrapped_structs
        name_bdf0, name_bdf1, name_udf0, name_udf1 = wrapped_names
        structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
        names_used = [name_udf0, name_udf1] if name_udf0 is not None else [name_bdf0, name_bdf1]

        # Naming
        pdb = names_used[0].split('.')[0]
        chain1 = names_used[0][-1]
        chain2 = names_used[1][-1]
        dirname = f"{pdb}_{chain1}{chain2}"
        dirpath = naming_utils.name_to_dir(dirname, dir_path=self.dump_dir)
        dump_path = os.path.join(dirpath, dirname)
        os.makedirs(dump_path, exist_ok=True)
        dump_struct_1 = os.path.join(dump_path, "struct_1.csv")
        dump_struct_2 = os.path.join(dump_path, "struct_2.csv")
        dump_pairs = os.path.join(dump_path, "pairs.csv")

        # Don't recompute for systems already loaded
        if not self.recompute and (os.path.exists(dump_struct_1) and
                                   os.path.exists(dump_struct_2) and
                                   os.path.exists(dump_pairs)):
            return 0, dirname, names_used[0], names_used[1]

        # If some surfaces are missing, skip the system
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
            return 1, None, None, None

        # Get all positives and negative neighbors, filter out non-empty hetero/insertion_code
        pos_neighbors_df = item['atoms_neighbors']
        non_heteros = []
        for df in structs_df:
            non_heteros.append(df[(df.hetero == ' ') & (df.insertion_code == ' ')].residue.unique())
        pos_neighbors_df = pos_neighbors_df[pos_neighbors_df.residue0.isin(non_heteros[0])
                                            & pos_neighbors_df.residue1.isin(non_heteros[1])]

        # First let us get all CA positions for both
        struct_1, struct_2 = structs_df
        struct_1 = struct_1[struct_1.residue.isin(non_heteros[0])]
        struct_1 = struct_1[struct_1.name == 'CA']
        struct_1 = struct_1[['chain', 'residue', 'resname', 'x', 'y', 'z', ]]
        struct_1 = struct_1.reset_index(drop=True)

        struct_2 = struct_2[struct_2.residue.isin(non_heteros[1])]
        struct_2 = struct_2[struct_2.name == 'CA']
        struct_2 = struct_2[['chain', 'residue', 'resname', 'x', 'y', 'z', ]]
        struct_2 = struct_2.reset_index(drop=True)

        # Now let us match the ids
        pos_pairs_res = pos_neighbors_df[['residue0', 'residue1']]

        # Finally dump all relevant files
        struct_1.to_csv(dump_struct_1)
        struct_2.to_csv(dump_struct_2)
        pos_pairs_res.to_csv(dump_pairs)
        return 0, dirname, names_used[0], names_used[1]

    def __getitem__(self, index):
        """

        :param index:
        :return: pos and neg arrays of the 2 partners CA 3D coordinates shape N_{pos,neg}x 2x 3
                 and the geometry objects necessary to embed the surfaces
        """
        try:
            # t0 = time.perf_counter()
            # self.original_getitem(index)
            # print("time load = ", time.perf_counter() - t0)

            ################
            # t0 = time.perf_counter()
            error_code, dirname, name1, name2 = self.reprocess(index)
            # print("time dump = ", time.perf_counter() - t0)
            return error_code, dirname, name1, name2
        except IndentationError as e:
            return 1, None, None, None


def reprocess_data(data_dir):
    dataset = PIPReprocess(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=6, collate_fn=lambda x: x[0])
    df = pd.DataFrame(columns=["system", "name1", "name2"])
    dump_csv = os.path.join(data_dir, 'all_systems.csv')
    # For now the time to beat is 0.2s to get the item
    for i, (error_code, dirname, name1, name2) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if error_code == 0:
            df.loc[len(df)] = dirname, name1, name2
        # if i > 5:
        #     break
    df.to_csv(dump_csv)





if __name__ == '__main__':
    data_dir = '../data/PIP/DIPS-split/data/test/'

    reprocess_data(data_dir)

    # dataset = NewPIP(data_dir)
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, collate_fn=lambda x: x[0])
    # for i, res in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    # # for i, res in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
    # #     print(i)
    #     if i > 5:
    #         break
