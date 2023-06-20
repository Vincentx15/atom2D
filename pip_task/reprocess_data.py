import os
import sys
import math

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
                 recompute_csv=False,
                 recompute_surfaces=False
                 ):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path, operator_path=operator_path)
        self.recompute = recompute_csv
        self.recompute_surfaces = recompute_surfaces
        self.dump_dir = os.path.join(os.path.dirname(lmdb_path), 'systems')

    def reprocess(self, index):
        item = self._lmdb_dataset[index]

        # Subunits
        wrapped_names, wrapped_structs = atom3dutils.get_subunits(item['atoms_pairs'])

        bdf0, bdf1, udf0, udf1 = wrapped_structs
        name_bdf0, name_bdf1, name_udf0, name_udf1 = wrapped_names
        structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
        names_used = [name_udf0, name_udf1] if name_udf0 is not None else [name_bdf0, name_bdf1]

        # if not (names_used[0] == '2a06.pdb1.gz_1_P' and
        #         names_used[1] == '2a06.pdb1.gz_1_T'):
        #     return 1, None, None, None

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

        if not self.recompute_surfaces:
            surface_0_exists = main.surface_exists(name=names_used[0],
                                                   dump_surf_dir=self.get_geometry_dir(names_used[0]),
                                                   dump_operator_dir=self.get_operator_dir(names_used[0]), )
            surface_1_exists = main.surface_exists(name=names_used[1],
                                                   dump_surf_dir=self.get_geometry_dir(names_used[1]),
                                                   dump_operator_dir=self.get_operator_dir(names_used[1]))
            # If some surfaces are missing, skip the system
            if not (surface_0_exists and surface_1_exists):
                return 1, None, None, None
        else:
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

            # If some surfaces are missing, skip the system
            if geom_feats_0 is None or geom_feats_1 is None:
                return 1, None, None, None

        # First let us get all CA positions for both
        def clean_struct(struct_df):
            """
            Filter out non-empty hetero/insertion_code, keep only CA, coordinates and resname
            """
            struct_df = struct_df[struct_df.element.isin(main.PROT_ATOMS)]
            struct_df = struct_df[(struct_df.hetero == ' ') & (struct_df.insertion_code == ' ')]
            struct_df = struct_df[['chain', 'residue', 'resname', 'name', 'element', 'x', 'y', 'z', ]]
            struct_df = struct_df.reset_index(drop=True)
            return struct_df

        struct_1, struct_2 = structs_df
        struct_1, struct_2 = clean_struct(struct_1), clean_struct(struct_2)

        # Get all positives and keep only valid ids to get a pair (no missing CA)
        ca_1 = struct_1[struct_1.name == 'CA']
        ca_2 = struct_2[struct_2.name == 'CA']
        res_1, res_2 = ca_1.residue, ca_2.residue
        pos_neighbors_df = item['atoms_neighbors']
        pos_neighbors_df_1 = pos_neighbors_df[pos_neighbors_df.residue0.isin(res_1)]
        pos_neighbors_df_2 = pos_neighbors_df_1[pos_neighbors_df_1.residue1.isin(res_2)]
        pos_pairs_res = pos_neighbors_df_2[['residue0', 'residue1']]

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
        except IndentationError:
            return 1, None, None, None


def reprocess_data(data_dir, recompute_csv=False, recompute_surfaces=False, num_workers=0):
    dataset = PIPReprocess(data_dir, recompute_csv=recompute_csv, recompute_surfaces=recompute_surfaces)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x[0])
    df = pd.DataFrame(columns=["system", "name1", "name2"])
    dump_csv = os.path.join(data_dir, 'all_systems.csv')
    # For now the time to beat is 0.2s to get the item
    for i, (error_code, dirname, name1, name2) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if error_code == 0:
            df.loc[len(df)] = dirname, name1, name2
        # if i > 5:
        #     break
    df.to_csv(dump_csv)


class NewPIP(torch.utils.data.Dataset):
    def __init__(self, data_dir, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=5,
                 geometry_path='../../data/processed_data/geometry/',
                 operator_path='../../data/processed_data/operator/',
                 return_graph=False,
                 recompute=False):
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.recompute = recompute
        csv_path = os.path.join(data_dir, 'all_systems.csv')
        self.dir_path = os.path.join(data_dir, 'systems')
        self.df = pd.read_csv(csv_path)
        self.geometry_path = geometry_path
        self.operator_path = operator_path
        self.return_graph = return_graph

    def get_geometry_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.geometry_path)

    def get_operator_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.operator_path)

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

    def __len__(self):
        return len(self.df)

    def get_item(self, index):
        # if not index == 42:
        #     return None
        row = self.df.iloc[index][["system", "name1", "name2"]]
        dirname, name1, name2 = row.values
        dirpath = naming_utils.name_to_dir(dirname, dir_path=self.dir_path)
        dump_path = os.path.join(dirpath, dirname)
        dump_struct_1 = os.path.join(dump_path, "struct_1.csv")
        dump_struct_2 = os.path.join(dump_path, "struct_2.csv")
        dump_pairs = os.path.join(dump_path, "pairs.csv")

        struct_1 = pd.read_csv(dump_struct_1, index_col=0)
        struct_2 = pd.read_csv(dump_struct_2, index_col=0)
        pos_pairs_res = pd.read_csv(dump_pairs, index_col=0)

        # Get CA coords
        ca_1 = struct_1[struct_1.name == 'CA']
        ca_2 = struct_2[struct_2.name == 'CA']
        mapping_1 = {resindex: i for i, resindex in enumerate(ca_1.residue.values)}
        mapping_2 = {resindex: i for i, resindex in enumerate(ca_2.residue.values)}

        pos_as_array_1 = np.array([mapping_1[resi] for resi in pos_pairs_res['residue0']])
        pos_as_array_2 = np.array([mapping_2[resi] for resi in pos_pairs_res['residue1']])

        dense = np.zeros((len(ca_1), len(ca_2)))
        dense[pos_as_array_1, pos_as_array_2] = 1
        negs_1, negs_2 = np.where(dense == 0)

        pos_array = np.stack((pos_as_array_1, pos_as_array_2))
        neg_array = np.stack((negs_1, negs_2))
        num_pos = pos_as_array_1.shape[0]
        num_neg = negs_1.shape[0]
        num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)
        pos_array_idx = np.random.choice(pos_array.shape[1], size=num_pos_to_use)
        neg_array_idx = np.random.choice(neg_array.shape[1], size=num_neg_to_use)
        pos_array_sampled = pos_array[:, pos_array_idx]
        neg_array_sampled = neg_array[:, neg_array_idx]

        # Get all coords, extract the right ones and stack them into (n_pairs, 2, 3)
        coords_1 = ca_1[['x', 'y', 'z', ]].values
        coords_2 = ca_2[['x', 'y', 'z', ]].values
        pos_1, pos_2 = coords_1[pos_array_sampled[0]], coords_2[pos_array_sampled[1]]
        pos_stack = np.transpose(np.stack((pos_1, pos_2)), axes=(1, 0, 2))
        neg_1, neg_2 = coords_1[neg_array_sampled[0]], coords_2[neg_array_sampled[1]]
        neg_stack = np.transpose(np.stack((neg_1, neg_2)), axes=(1, 0, 2))

        geom_feats_0 = main.get_diffnetfiles(name=name1,
                                             df=struct_1,
                                             dump_surf_dir=self.get_geometry_dir(name1),
                                             dump_operator_dir=self.get_operator_dir(name1),
                                             recompute=self.recompute)
        geom_feats_1 = main.get_diffnetfiles(name=name2,
                                             df=struct_2,
                                             dump_surf_dir=self.get_geometry_dir(name2),
                                             dump_operator_dir=self.get_operator_dir(name2),
                                             recompute=self.recompute)

        if geom_feats_0 is None or geom_feats_1 is None:
            return None, None, None, None, None, None
        return name1, name2, torch.from_numpy(pos_stack), torch.from_numpy(neg_stack), geom_feats_0, geom_feats_1

    def __getitem__(self, index):
        # res = self.get_item(index)
        try:
            res = self.get_item(index)
            return res
        except Exception as e:
            print(f"Error in __getitem__: {e} index : {index}")
            return None, None, None, None, None, None


if __name__ == '__main__':
    data_dir = '../data/PIP/DIPS-split/data/train/'
    reprocess_data(data_dir, recompute_csv=True, num_workers=4)
    data_dir = '../data/PIP/DIPS-split/data/val/'
    reprocess_data(data_dir, recompute_csv=True, num_workers=4)
    data_dir = '../data/PIP/DIPS-split/data/test/'
    reprocess_data(data_dir, recompute_csv=True, num_workers=4)

    # import time
    # t0 = time.perf_counter()
    # dataset = NewPIP(data_dir)
    # dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, collate_fn=lambda x: x[0])
    # for i, res in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
    #     # for i, res in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
    #     # print(i)
    #     if i > 250:
    #         break
    # print(time.perf_counter() - t0)
