import numpy as np
import os
import pandas as pd
import subprocess
import time
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_net import geometry
from atom3d.datasets import LMDBDataset

from df_utils import df_to_pdb

"""
In this file, we define functions to make the following transformations : 
PDB -> surfaces in .vert+.faces -> DiffNets operators in .npz format

We also define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to 
"""


def pdb_to_surf(pdb, out_name):
    """
    Runs msms on the input PDB file and dumps the output in out_name
    :param pdb:
    :param out_name:
    :return:
    """
    # First get the xyzr file
    temp_xyzr_name = f"{out_name}_temp.xyzr"
    temp_log_name = f"{out_name}_msms.log"
    with open(temp_xyzr_name, "w") as f:
        cline = f"pdb_to_xyzr {pdb}"
        subprocess.run(cline.split(), stdout=f)
    cline = f"msms -if {temp_xyzr_name} -of {out_name}"
    with open(temp_log_name, "w") as f:
        subprocess.run(cline.split(), stdout=f)
    os.remove(temp_xyzr_name)
    os.remove(temp_log_name)
    pass


def surf_to_operators(vert_file, face_file, dump_dir, recompute=False):
    """
    Takes the output of msms and dump the diffusion nets operators in dump dir
    :param vert_file:
    :param face_file:
    :param dump_dir:
    :return:
    """
    with open(vert_file, 'r') as f:
        # Parse the file and ensure it looks sound
        lines = f.readlines()
        n_vert = int(lines[2].split()[0])
        no_header = lines[3:]
        assert len(no_header) == n_vert

        # Parse the info to retrieve vertices and normals
        lines = [line.split() for line in no_header]
        lines = np.array(lines).astype(np.float)
        verts = lines[:, :3]
        # normals = lines[:, 3:6]

    with open(face_file, 'r') as f:
        # Parse the file and ensure it looks sound
        lines = f.readlines()
        n_faces = int(lines[2].split()[0])
        no_header = lines[3:]
        assert len(no_header) == n_faces

        # Parse the lines and remove 1 to get zero based indexing
        lines = [line.split() for line in no_header]
        lines = np.array(lines).astype(np.int)
        faces = lines[:, :3]
        faces -= 1

    verts = torch.from_numpy(np.ascontiguousarray(verts))
    faces = torch.from_numpy(np.ascontiguousarray(faces))
    # pre_normals = torch.from_numpy(np.ascontiguousarray(normals))
    normals = None

    print(f'found {len(verts)} vertices')
    # print(verts.shape)
    # print(faces.shape)

    frames, mass, L, evals, evecs, gradX, gradY = geometry.get_operators(verts=verts,
                                                                         normals=normals,
                                                                         faces=faces,
                                                                         op_cache_dir=dump_dir,
                                                                         overwrite_cache=recompute)

    # computed_normals = frames[:, 2, :]
    # print(computed_normals.shape)
    # print(pre_normals.shape)
    # print(pre_normals[0])
    # print(computed_normals[0])
    # print(torch.dot(pre_normals[0], computed_normals[0]))
    # print(torch.allclose(computed_normals, pre_normals))
    return frames, mass, L, evals, evecs, gradX, gradY


def process_df(df, dump_surf, dump_operator, recompute=False):
    """

    :param df: a df that represents a protein
    :param dump_surf: the basename of the surface .vert and .faces to compute
    :param dump_operator: The dir where diffusion net searches for precomputed data
    :return:
    """
    # if they are missing, compute the surface from the df
    vert_file = dump_surf + '.vert'
    face_file = dump_surf + '.face'
    if (not os.path.exists(vert_file) or not os.path.exists(face_file)) or recompute:
        t_0 = time.perf_counter()
        temp_pdb = dump_surf + '.pdb'
        df_to_pdb(df, out_file_name=temp_pdb)
        pdb_to_surf(temp_pdb, out_name=dump_surf)
        os.remove(temp_pdb)
        print('time to process msms : ', time.perf_counter() - t_0)
    if not os.path.exists(dump_operator) or recompute:
        t_0 = time.perf_counter()
        operators = surf_to_operators(vert_file, face_file, dump_dir=dump_operator, recompute=recompute)
        print('time to process diffnets : ', time.perf_counter() - t_0)
    return


def get_subunits(ensemble):
    subunits = ensemble['subunit'].unique()

    if len(subunits) == 4:
        lb = [x for x in subunits if x.endswith('ligand_bound')][0]
        lu = [x for x in subunits if x.endswith('ligand_unbound')][0]
        rb = [x for x in subunits if x.endswith('receptor_bound')][0]
        ru = [x for x in subunits if x.endswith('receptor_unbound')][0]
        bdf0 = ensemble[ensemble['subunit'] == lb]
        bdf1 = ensemble[ensemble['subunit'] == rb]
        udf0 = ensemble[ensemble['subunit'] == lu]
        udf1 = ensemble[ensemble['subunit'] == ru]
        names = (lb, rb, lu, ru)
    elif len(subunits) == 2:
        udf0, udf1 = None, None
        bdf0 = ensemble[ensemble['subunit'] == subunits[0]]
        bdf1 = ensemble[ensemble['subunit'] == subunits[1]]
        names = (subunits[0], subunits[1], None, None)
    else:
        raise RuntimeError('Incorrect number of subunits for pair')
    return names, (bdf0, bdf1, udf0, udf1)


class MapAtom3DDataset(Dataset):
    def __init__(self, lmdb_path):
        _lmdb_dataset = LMDBDataset(lmdb_path)
        self.lenght = len(_lmdb_dataset)
        self._lmdb_dataset = None
        self.lmdb_path = lmdb_path

    def __len__(self) -> int:
        return self.lenght

    def __getitem__(self, index):

        if self._lmdb_dataset is None:
            self._lmdb_dataset = LMDBDataset(self.lmdb_path)
        item = self._lmdb_dataset[index]
        # Subunits
        names, (bdf0, bdf1, udf0, udf1) = get_subunits(item['atoms_pairs'])

        structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
        # Throw away non empty hetero/insertion_code
        non_heteros = []
        for df in structs_df:
            non_heteros.extend(df[(df.hetero == ' ') & (df.insertion_code == ' ')].residue.unique())
        filtered_df = []
        for df in structs_df:
            filtered_df.append(df[df.residue.isin(non_heteros)])

        for name, dataframe in zip(names, filtered_df):
            if name is None:
                continue
            else:
                print(name)
                process_df(df=dataframe,
                           dump_surf=f'data/processed_data/geometry/{name}',
                           dump_operator='data/processed_data/operator/',
                           recompute=False)
                print()
        return 1


def collate_fn(samples):
    """
    A non op to avoid torch casting as we only use it for the multiprocessing here (and inheritance from LMDB)
    :param samples:
    :return:
    """
    return samples


# Finally, we need to iterate to precompute all relevant surfaces and operators
def compute_operators_all(data_dir):
    train_dataset = MapAtom3DDataset(data_dir)
    loader = torch.utils.data.DataLoader(train_dataset, num_workers=1, batch_size=1, collate_fn=collate_fn)
    # all_names = set()
    for i, names in enumerate(loader):
        # for name in names[0]:
        #     all_names.add(name)
        # break
        if i > 50:
            break
        # if not i % 100:
        # print()
        # print(f"{i}/{len(loader)} processed")
        # print(f"Discoverd {len(all_names)} pdb")
        # 87300/87303 processed
        # Discoverd 108805 pdb


if __name__ == '__main__':
    # pdb_to_surf(pdb='data/example_files/from_biopython.pdb', out_name='data/example_files/test')
    # pdb_to_surf(pdb='data/example_files/from_db.pdb', out_name='data/example_files/test')
    # surf_to_operators(vert_file='data/example_files/test.vert',
    #                   face_file='data/example_files/test.face',
    #                   dump_dir='data/processed_data/operators')

    df = pd.read_csv('data/example_files/4kt3.csv')

    # np.random.seed(0)
    # torch.manual_seed(0)

    # process_df(df=df,
    #            dump_surf='data/processed_data/geometry/4kt3',
    #            dump_operator='data/processed_data/operator/')
    compute_operators_all(data_dir='data/DIPS-split/data/train')
