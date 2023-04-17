import os
import sys

import numpy as np

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing import df_utils, point_cloud_utils, surface_utils, get_operators


def process_df(df, name, dump_surf_dir, dump_operator_dir, recompute=False, min_number=128 * 4, max_error=5):
    """
    The whole process of data creation, from df format of atom3D to ply files and precomputed operators.

    We have to get enough points on the surface to avoid eigen decomposition problems, so we potentially resample
    over the surface using msms.
    Then we simplify all meshes to get closer to this value, up to a certain error, using open3d coarsening.

    Without coarsening, time is dominated by eigen decomposition and is approx 5s. Operator files weigh around 10M,
        small ones around 400k, big ones up to 16M
    With a max_error of 5, time is dominated by MSMS and is close to ones. Operator files weigh around 0.6M, small ones
        around 400k, big ones up to 1.1M

    :param df: a df that represents a protein
    :param name: Name of the corresponding systems
    :param dump_surf_dir: the basename of the surface to dump
    :param dump_operator_dir: The dir where diffusion net searches for precomputed data
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

    # Need recomputing ?
    ply_ex = os.path.exists(ply_file)
    feat_ex = os.path.exists(features_file)
    ope_ex = os.path.exists(dump_operator_file)
    if not os.path.exists(ply_file) or not os.path.exists(features_file) or not os.path.exists(dump_operator_file):
        print(f'Precomputing {name}, ply : {ply_ex}, feat : {feat_ex}, ope : {ope_ex}')

    # Get pdb file only if needed
    need_pdb = recompute or not os.path.exists(ply_file) or not os.path.exists(features_file)
    if need_pdb:
        df_utils.df_to_pdb(df, out_file_name=temp_pdb)

    # if they are missing, compute the surface from the df. Get a temp PDB, parse it with msms and simplify it
    if not os.path.exists(ply_file) or recompute:
        # t_0 = time.perf_counter()
        vert_file = dump_surf + '.vert'
        face_file = dump_surf + '.face'
        surface_utils.pdb_to_surf_with_min(temp_pdb, out_name=dump_surf, min_number=min_number)
        mesh = surface_utils.mesh_simplification(vert_file=vert_file,
                                                 face_file=face_file,
                                                 out_ply=ply_file,
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

    if need_pdb:
        os.remove(temp_pdb)

    # t_0 = time.perf_counter()

    if not os.path.exists(dump_operator_file) or recompute:
        if vertices is None or faces is None:
            vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
        get_operators.surf_to_operators(vertices=vertices, faces=faces, npz_path=dump_operator_file,
                                        recompute=recompute)
    # print('time to process diffnets : ', time.perf_counter() - t_0)


if __name__ == '__main__':
    pass
    import pandas as pd

    root_dir = '../data/example_files/'
    df_path = os.path.join(root_dir, '4kt3.csv')
    pdb_path = os.path.join(root_dir, '4kt3.pdb')

    # pdb -> ply
    df = pd.read_csv(df_path, keep_default_na=False)
    df_utils.df_to_pdb(df, pdb_path)
    out_name = os.path.join(root_dir, '4kt3_mesh')
    vert_file = f"{out_name}.vert"
    faces_file = f"{out_name}.face"
    ply_file = os.path.join(root_dir, '4kt3_mesh.ply')

    surface_utils.pdb_to_surf_with_min(pdb=pdb_path, out_name=out_name)
    surface_utils.mesh_simplification(vert_file=vert_file,
                                      face_file=faces_file,
                                      out_ply=ply_file,
                                      vert_number=1000,
                                      maximum_error=5)

    # ply -> operators + features
    features_file = os.path.join(root_dir, '4kt3_features.npz')
    dump_operator_file = os.path.join(root_dir, "4kt3_operator.npz")
    vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
    get_operators.surf_to_operators(vertices=vertices, faces=faces,
                                    npz_path=dump_operator_file)
    features, confidence = point_cloud_utils.get_features(pdb_path, vertices)
    np.savez_compressed(features_file, **{'features': features, 'confidence': confidence})

    # All at once
    process_df(df=df,
               name='4kt3',
               dump_surf_dir='../data/example_files/4kt3',
               dump_operator_dir='../data/example_files/4kt3',
               recompute=True)
