import os
import sys

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


if __name__ == '__main__':
    data_dir = '../data/PIP/DIPS-split/data/train/'
    reprocess_data(data_dir, recompute_csv=True, num_workers=4)
    data_dir = '../data/PIP/DIPS-split/data/val/'
    reprocess_data(data_dir, recompute_csv=True, num_workers=4)
    data_dir = '../data/PIP/DIPS-split/data/test/'
    reprocess_data(data_dir, recompute_csv=True, num_workers=4)
