import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data

import atom3d.util.graph as gr

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, ".."))

from data_processing import df_utils, point_cloud_utils, surface_utils, get_operators
from atom2d_utils import learning_utils


def process_df(df, name, dump_surf_dir, dump_operator_dir, recompute=False, min_number=2000, verbose=False,
               clean_temp=True):
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
    :param clean_temp: To remove temp files such as yzr files. This can cause bugs when processing the same pdb in
    parallel for different chains for instance
    :return:
    """
    # Optionnally setup dirs
    is_valid_mesh = True
    os.makedirs(dump_surf_dir, exist_ok=True)
    os.makedirs(dump_operator_dir, exist_ok=True)
    dump_surf = os.path.join(dump_surf_dir, name)
    temp_pdb = dump_surf + ".pdb"
    ply_file = f"{dump_surf}_mesh.ply"
    features_file = f"{dump_surf}_features.npz"
    vertices, faces = None, None
    dump_operator_file = os.path.join(dump_operator_dir, f"{name}_operator.npz")

    if verbose:
        print(f"Processing {name}")

    # Need recomputing ?
    ply_ex = os.path.exists(ply_file)
    feat_ex = os.path.exists(features_file)
    ope_ex = os.path.exists(dump_operator_file)
    need_recompute = not os.path.exists(ply_file) or \
                     not os.path.exists(features_file) or \
                     not os.path.exists(dump_operator_file)
    if not (need_recompute or recompute):
        return is_valid_mesh
    if verbose:
        if recompute:
            print(f"Recomputing {name}")
        else:
            print(f"Precomputing {name}, found ply : {ply_ex}, found feat : {feat_ex}, found ope : {ope_ex}")

    # Get pdb file only if needed
    need_pdb = recompute or not os.path.exists(ply_file) or not os.path.exists(features_file)
    if need_pdb:
        if verbose:
            print(f"Converting df to pdb for {name}")
        df_utils.df_to_pdb(df, out_file_name=temp_pdb)

    # if they are missing, compute the surface from the df. Get a temp PDB, parse it with msms and simplify it
    if not os.path.exists(ply_file) or recompute:
        if verbose:
            print(f"Computing surface for {name} with msms")
        vertices, faces = surface_utils.pdb_to_surf_with_min(temp_pdb, out_name=dump_surf, min_number=min_number,
                                                             clean_temp=clean_temp)

        if verbose:
            print(f"Simplifying surface for {name}")
        vertices, faces, is_valid_mesh = surface_utils.mesh_simplification(verts=vertices, faces=faces,
                                                                           out_ply=ply_file, vert_number=min_number)

    if not os.path.exists(features_file) or recompute:
        if verbose:
            print(f"Computing features for {name}")
        if vertices is None or faces is None:
            vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
        features, confidence = point_cloud_utils.get_features(temp_pdb, vertices)
        np.savez_compressed(features_file, **{"features": features, "confidence": confidence})

    if need_pdb and clean_temp:
        os.remove(temp_pdb)

    if not os.path.exists(dump_operator_file) or recompute:
        if verbose:
            print(f"Computing operators for {name}")
        if vertices is None or faces is None:
            vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
        operators = get_operators.surf_to_operators(vertices=vertices, faces=faces, npz_path=dump_operator_file,
                                                    recompute=recompute)

    return is_valid_mesh


def get_diffnetfiles(name, df, dump_surf_dir, dump_operator_dir, recompute=True):
    """
    Get all relevant files, potentially recomputing them as needed
    :param name: The name of the queried file
    :param df: The corresponding dataframe
    :param dump_surf_dir: Where to store the geometry (ply)
    :param dump_operator_dir: Where to store the operators
    :param recompute: If file is missing, shall we recompute ?
    :return:
    """
    dump_surf_outname = os.path.join(dump_surf_dir, name)
    ply_file = f"{dump_surf_outname}_mesh.ply"
    features_file = f"{dump_surf_outname}_features.npz"
    operator_file = f"{dump_operator_dir}/{name}_operator.npz"

    need_recompute = not (os.path.exists(ply_file) and os.path.exists(features_file) and os.path.exists(operator_file))
    if need_recompute:
        if recompute:
            print(
                f"For system : {name}, recomputing : "
                f"{'geometry, ' if not os.path.exists(ply_file) else ''}"
                f"{'features, ' if not os.path.exists(features_file) else ''}"
                f"{'operator, ' if not os.path.exists(operator_file) else ''}"
            )
            process_df(df=df, name=name, dump_surf_dir=dump_surf_dir, dump_operator_dir=dump_operator_dir)
        else:
            return None
            # raise FileNotFoundError("The precomputed file could not be found for ", name)

    vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
    features_dump = np.load(features_file)
    features, confidence = features_dump["features"], features_dump["confidence"]
    vertices, faces, features, confidence = learning_utils.list_from_numpy([vertices, faces, features, confidence])
    frames, mass, _, evals, evecs, grad_x, grad_y = get_operators.surf_to_operators(vertices=vertices,
                                                                                    faces=faces,
                                                                                    npz_path=operator_file)
    return features, confidence, vertices, mass, torch.rand(1,
                                                            3), evals, evecs, grad_x.to_dense(), grad_y.to_dense(), faces


# prot_atoms = ['C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO', 'MG', 'CU', 'CL', 'SE', 'F']
prot_atoms = ['C', 'O', 'N', 'S', 'X']

import scipy.spatial as ss
from torch_geometric.utils import to_undirected


def one_of_k_encoding_unk(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values.
     Additionally maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def prot_df_to_graph(df, feat_col='element', allowable_feats=prot_atoms, edge_dist_cutoff=4.5):
    r"""
    Converts protein in dataframe representation to a graph compatible with Pytorch-Geometric, where each node is an atom.

    :param df: Protein structure in dataframe format.
    :type df: pandas.DataFrame
    :param node_col: Column of dataframe to find node feature values. For example, for atoms use ``feat_col="element"`` and for residues use ``feat_col="resname"``
    :type node_col: str, optional
    :param allowable_feats: List containing all possible values of node type, to be converted into 1-hot node features.
        Any elements in ``feat_col`` that are not found in ``allowable_feats`` will be added to an appended "unknown" bin (see :func:`atom3d.util.graph.one_of_k_encoding_unk`).
    :type allowable_feats: list, optional
    :param edge_dist_cutoff: Maximum distance cutoff (in Angstroms) to define an edge between two atoms, defaults to 4.5.
    :type edge_dist_cutoff: float, optional

    :return: tuple containing

        - node_feats (torch.FloatTensor): Features for each node, one-hot encoded by values in ``allowable_feats``.

        - edges (torch.LongTensor): Edges in COO format

        - edge_weights (torch.LongTensor): Edge weights, defined as a function of distance between atoms given by :math:`w_{i,j} = \frac{1}{d(i,j)}`, where :math:`d(i, j)` is the Euclidean distance between node :math:`i` and node :math:`j`.

        - node_pos (torch.FloatTensor): x-y-z coordinates of each node
    :rtype: Tuple
    """
    # import time
    # t0 = time.time()
    df = df.loc[df['element'] != 'H']  # TODO : maybe consider doing all atom, as they DO NOT do this operation.
    # TODO : I added it for speed.
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())
    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()
    edges = to_undirected(edges)
    node_feats = torch.FloatTensor([one_of_k_encoding_unk(e, allowable_feats) for e in df[feat_col]])
    # print(f"time to pre_dist : {time.time() - t0}")

    # t0 = time.time()
    node_a = node_pos[edges[0, :]]
    node_b = node_pos[edges[1, :]]
    with torch.no_grad():
        my_edge_weights_torch = 1 / (torch.linalg.norm(node_a - node_b, axis=1) + 1e-5)
    # print(f"time to torch : {time.time() - t0}")
    # t0 = time.time()
    # edge_weights = torch.FloatTensor(
    #     [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edges.t()]).view(-1)
    # print(f"time to do their : {time.time() - t0}")
    # error_torch = (my_edge_weights_torch - edge_weights).max()
    # feats = F.one_hot(elems, num_classes=len(atom_int_dict))
    # print(f"errors, torch : {error_torch}")

    return node_feats, edges, my_edge_weights_torch, node_pos


def get_graph(name, df, dump_graph_dir, xyz=None, recompute=False):
    os.makedirs(dump_graph_dir, exist_ok=True)
    dump_graph_name = os.path.join(dump_graph_dir, f"{name}.pth")
    if not os.path.exists(dump_graph_name):
        if recompute:
            print(
                f"For system : {name}, recomputing : graph "
            )
            node_feats, edge_index, edge_feats, pos = prot_df_to_graph(df, allowable_feats=prot_atoms)

            # !!! not sure if this needed, cdist seems to be quick
            distance_matrix = torch.cdist(xyz, pos)
            N, M = distance_matrix.shape
            k, sigma = 30, 2.5

            _, knn_indices = torch.topk(distance_matrix, k, largest=False, sorted=True, dim=1)
            row_indices = torch.arange(N).view(-1, 1).expand(-1, k).reshape(-1)
            knn_distances = torch.gather(distance_matrix, 1, knn_indices)
            rbf_weight_values = torch.exp(-knn_distances / sigma).reshape(-1)
            indices = torch.stack((row_indices, knn_indices.reshape(-1)))
            rbf_surf_graph = torch.sparse.FloatTensor(indices, rbf_weight_values, (N, B.size(0)))

            _, knn_indices = torch.topk(distance_matrix.T, k, largest=False, sorted=True, dim=1)
            row_indices = torch.arange(M).view(-1, 1).expand(-1, k).reshape(-1)
            knn_distances = torch.gather(distance_matrix.T, 1, knn_indices)
            rbf_weight_values = torch.exp(-knn_distances / sigma).reshape(-1)
            indices = torch.stack((row_indices, knn_indices.reshape(-1)))
            rbf_graph_surf = torch.sparse.FloatTensor(indices, rbf_weight_values, (N, B.size(0)))

            graph = Data(node_feats, edge_index, edge_feats, pos=pos, rbf_surf_graph=rbf_surf_graph, rbf_graph_surf=rbf_graph_surf)
            graph = Data(node_feats, edge_index, edge_feats, pos=pos)
            torch.save(graph, dump_graph_name)
        else:
            return None
            # raise FileNotFoundError("The precomputed file could not be found for ", name)
    graph = torch.load(dump_graph_name)
    return graph


if __name__ == "__main__":
    pass
    import pandas as pd

    root_dir = "../data/example_files/"
    df_path = os.path.join(root_dir, "4kt3.csv")
    pdb_path = os.path.join(root_dir, "4kt3.pdb")

    # pdb -> ply
    df = pd.read_csv(df_path, keep_default_na=False)
    df_utils.df_to_pdb(df, pdb_path)
    out_name = os.path.join(root_dir, "4kt3_mesh")
    ply_file = os.path.join(root_dir, "4kt3_mesh.ply")

    verts, faces = surface_utils.pdb_to_surf_with_min(pdb=pdb_path, out_name=out_name)
    surface_utils.mesh_simplification(verts=verts, faces=faces, out_ply=ply_file, vert_number=1000)

    # ply -> operators + features
    features_file = os.path.join(root_dir, "4kt3_features.npz")
    dump_operator_file = os.path.join(root_dir, "4kt3_operator.npz")
    vertices, faces = surface_utils.read_vertices_and_triangles(ply_file=ply_file)
    get_operators.surf_to_operators(vertices=vertices, faces=faces, npz_path=dump_operator_file)
    features, confidence = point_cloud_utils.get_features(pdb_path, vertices)
    np.savez_compressed(features_file, **{"features": features, "confidence": confidence})

    # All at once
    process_df(df=df, name="4kt3",
               dump_surf_dir="../data/example_files/4kt3",
               dump_operator_dir="../data/example_files/4kt3",
               recompute=True)
