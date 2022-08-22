import sys

import numpy as np
import os
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader, Dataset

from atom3d.datasets import LMDBDataset

from atom3dutils import get_subunits
import df_utils
import point_cloud_utils
import surface_utils
import get_operators
import utils

"""
Here, we define a way to iterate through a .mdb file as defined by ATOM3D
and leverage PyTorch parallel data loading to efficiently do this preprocessing
"""


def process_df(df, name, dump_surf_dir, dump_operator_dir, recompute=False, min_number=128 * 4, max_error=5):
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
    # Optionnally setup dirs
    os.makedirs(dump_surf_dir, exist_ok=True)
    os.makedirs(dump_operator_dir, exist_ok=True)
    dump_surf = os.path.join(dump_surf_dir, name)
    dump_operator = os.path.join(dump_operator_dir, name)

    temp_pdb = dump_surf + '.pdb'
    ply_file = f"{dump_surf}_mesh.ply"
    features_file = f"{dump_surf}_features.npz"
    vertices, faces = None, None
    dump_operator_file = f"{dump_operator}_operator.npz"

    # Get pdb file
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
        vertices, faces = surface_utils.get_vertices_and_triangles(mesh)

        # print('time to process msms and simplify mesh: ', time.perf_counter() - t_0)
        os.remove(vert_file)
        os.remove(face_file)

    # t_0 = time.perf_counter()
    if not os.path.exists(features_file) or recompute:
        if vertices is None or faces is None:
            vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
        features, confidence = point_cloud_utils.get_features(temp_pdb, vertices)
        np.savez_compressed(features_file, **{'features': features, 'confidence': confidence})
    # print('time get_features: ', time.perf_counter() - t_0)
    os.remove(temp_pdb)

    # t_0 = time.perf_counter()

    if not os.path.exists(dump_operator_file) or recompute:
        if vertices is None or faces is None:
            vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
        get_operators.surf_to_operators(vertices=vertices, faces=faces, npz_path=dump_operator_file,
                                        recompute=recompute)
    # print('time to process diffnets : ', time.perf_counter() - t_0)
    return


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
        # names : ('117e.pdb1.gz_1_A', '117e.pdb1.gz_1_B', None, None)
        names, (bdf0, bdf1, udf0, udf1) = get_subunits(item['atoms_pairs'])

        # For pinpointing one pdb code that would be buggy
        # for name in names:
        #     if not "1jcc.pdb1.gz_1_C" in name:
        #         return
        # print('doing a buggy one')

        structs_df = [udf0, udf1] if udf0 is not None else [bdf0, bdf1]
        for name, dataframe in zip(names, structs_df):
            if name is None:
                continue
            else:
                if name in self.failed_set:
                    return 0
                try:
                    dump_surf_dir = os.path.join('data/processed_data/geometry/', utils.name_to_dir(name))
                    dump_operator_dir = os.path.join('data/processed_data/operator/', utils.name_to_dir(name))
                    process_df(df=dataframe,
                               name=name,
                               dump_surf_dir=dump_surf_dir,
                               dump_operator_dir=dump_operator_dir,
                               recompute=False)
                    # print(f'Precomputed successfully for {name}')
                except:
                    self.failed_set.add(name)
                    # print(f'Failed precomputing for {name}')
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
    t0 = time.time()
    train_dataset = MapAtom3DDataset(data_dir)
    train_dataset = torch.utils.data.DataLoader(train_dataset, num_workers=os.cpu_count(), batch_size=1,
                                                collate_fn=collate_fn)
    for i, success in enumerate(train_dataset):
        pass
        if not i % 100:
            print(f"Done {i} in {time.time() - t0}")
        # if i > 0:
        #     break


if __name__ == '__main__':
    pass
    # pdb_to_surf(pdb='data/example_files/from_biopython.pdb', out_name='data/example_files/test')
    # pdb_to_surf(pdb='data/example_files/from_db.pdb', out_name='data/example_files/test')
    # surf_to_operators(vert_file='data/example_files/test.vert',
    #                   face_file='data/example_files/test.face',
    #                   dump_dir='data/processed_data/operators')

    np.random.seed(0)
    torch.manual_seed(0)

    # df = pd.read_csv('data/example_files/4kt3.csv')
    # process_df(df=df,
    #            dump_surf='data/example_files/4kt3',
    #            dump_operator='data/example_files/4kt3')
    compute_operators_all(data_dir='data/DIPS-split/data/train/')

    # A first run gave us 100k pdb in the DB.
    # 87300/87303 processed
    # Discoverd 108805 pdb

# 1jcc.pdb1.gz_1_C SEGFAULT ?
