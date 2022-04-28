import numpy as np
import os
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader, Dataset

from diffusion_net import geometry
from atom3d.datasets import LMDBDataset

import df_utils
import point_cloud_utils
import surface_utils
import utils

"""
In this file, we define functions to make the following transformations : 
.ply -> DiffNets operators in .npz format

We also define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to 
"""


def surf_to_operators(vertices, faces, dump_dir, recompute=False):
    """
    Takes the output of msms and dump the diffusion nets operators in dump dir
    :param vert_file: Vx3 tensor of coordinates
    :param face_file: Fx3 tensor of indexes, zero based
    :param dump_dir: Where to dump the precomputed operators
    :return:
    """

    verts = torch.from_numpy(np.ascontiguousarray(vertices))
    faces = torch.from_numpy(np.ascontiguousarray(faces))
    frames, mass, L, evals, evecs, gradX, gradY = geometry.get_operators(verts=verts,
                                                                         faces=faces,
                                                                         op_cache_dir=dump_dir,
                                                                         overwrite_cache=recompute)

    # pre_normals = torch.from_numpy(np.ascontiguousarray(normals))
    # computed_normals = frames[:, 2, :]
    # print(computed_normals.shape)
    # print(pre_normals.shape)
    # print(pre_normals[0])
    # print(computed_normals[0])
    # print(torch.dot(pre_normals[0], computed_normals[0]))
    # print(torch.allclose(computed_normals, pre_normals))
    return frames, mass, L, evals, evecs, gradX, gradY


def process_df(df, dump_surf, dump_operator, recompute=False, min_number=128 * 4, max_error=5):
    """
    The whole process of data creation, from df format of atom3D to ply files and precomputed operators.

    We have to get enough points on the surface to avoid eigendecomposition problems, so we potentially resample
    over the surface using msms. Then we simplify all meshes to get closer to this value, up to a certain error,
    using open3d coarsening.

    Without coarsening, time is dominated by eigendecomposition and is approx 5s. Operator files weigh around 10M, small
        ones around 400k, big ones up to 16M
    With a max_error of 5, time is dominated by MSMS and is close to ones. Operator files weigh around 0.6M, small ones
        around 400k, big ones up to 1.1M

    :param df: a df that represents a protein
    :param dump_surf: the basename of the surface to dump
    :param dump_operator: The dir where diffusion net searches for precomputed data
    :param recompute: to force recomputation of cached files
    :param min_number: The minimum number of points of the final mesh, we take 4 times the size of kept eigenvalues
    :param max_error: The maximum error when coarsening the mesh
    :return:
    """
    ply_file = f"{dump_surf}_mesh.ply"
    features_file = f"{dump_surf}_features.npy"
    temp_pdb = dump_surf + '.pdb'
    df_utils.df_to_pdb(df, out_file_name=temp_pdb)

    # if they are missing, compute the surface from the df. Get a temp PDB, parse it with msms and simplify it
    if not os.path.exists(ply_file) or recompute:
        # t_0 = time.perf_counter()
        vert_file = dump_surf + '.vert'
        face_file = dump_surf + '.face'
        surface_utils.pdb_to_surf_with_min(temp_pdb, out_name=dump_surf, min_number=min_number)
        mesh = surface_utils.mesh_simplification(vert_file=vert_file,
                                                 face_file=face_file,
                                                 out_name=dump_surf,
                                                 vert_number=min_number,
                                                 maximum_error=max_error)
        # print('time to process msms and simplify mesh: ', time.perf_counter() - t_0)
        os.remove(vert_file)
        os.remove(face_file)

    vertices, faces = surface_utils.read_face_and_triangles(ply_file=ply_file)
    # t_0 = time.perf_counter()
    if not os.path.exists(features_file) or recompute:
        features = point_cloud_utils.get_features(temp_pdb, vertices)
        np.save(features_file, features)
    # print('time get_features: ', time.perf_counter() - t_0)
    os.remove(temp_pdb)


    # t_0 = time.perf_counter()
    operators = surf_to_operators(vertices=vertices, faces=faces, dump_dir=dump_operator, recompute=recompute)
    # print('time to process diffnets : ', time.perf_counter() - t_0)
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
        self.failed_set = set()
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
                if name in self.failed_set:
                    return 0
                try:
                    dump_surf_dir = os.path.join('data/processed_data/geometry/', utils.name_to_dir(name))
                    dump_surf_outname = os.path.join(dump_surf_dir, name)
                    dump_operator = os.path.join('data/processed_data/operator/', utils.name_to_dir(name))
                    os.makedirs(dump_surf_dir, exist_ok=True)
                    os.makedirs(dump_operator, exist_ok=True)
                    process_df(df=dataframe,
                               dump_surf=dump_surf_outname,
                               dump_operator=dump_operator,
                               recompute=False)
                    print(f'Precomputed successfully for {name}')
                except:
                    self.failed_set.add(name)
                    print(f'Failed precomputing for {name}')
                    return 0
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
    loader = torch.utils.data.DataLoader(train_dataset, num_workers=6, batch_size=1, collate_fn=collate_fn)
    for i, success in enumerate(loader):
        if i > 50:
            break


if __name__ == '__main__':
    pass
    # pdb_to_surf(pdb='data/example_files/from_biopython.pdb', out_name='data/example_files/test')
    # pdb_to_surf(pdb='data/example_files/from_db.pdb', out_name='data/example_files/test')
    # surf_to_operators(vert_file='data/example_files/test.vert',
    #                   face_file='data/example_files/test.face',
    #                   dump_dir='data/processed_data/operators')

    # np.random.seed(0)
    # torch.manual_seed(0)

    df = pd.read_csv('data/example_files/4kt3.csv')
    process_df(df=df,
               dump_surf='data/example_files/4kt3',
               dump_operator='data/example_files/4kt3')
    # compute_operators_all(data_dir='data/DIPS-split/data/train/')

    # A first run gave us 100k pdb in the DB.
    # 87300/87303 processed
    # Discoverd 108805 pdb
