import os
import sys

import numpy as np
import open3d as o3d
import torch
from torch_geometric.data import Data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, ".."))

from data_processing import get_operators
from atom2d_utils import learning_utils


def load_graph(name, dump_graph_dir):
    dump_graph_name = os.path.join(dump_graph_dir, f"{name}.pth")
    if not os.path.exists(dump_graph_name):
        return None
    graph = torch.load(dump_graph_name)
    return graph


def read_vertices_and_triangles(ply_file):
    """
    Just a small wrapper to retrieve directly the vertices and faces as np arrays with the right dtypes
    :param ply_file:
    :return:
    """

    mesh = o3d.io.read_triangle_mesh(filename=ply_file)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, faces


def load_diffnetfiles(name, dump_surf_dir, dump_operator_dir):
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

    if not os.path.exists(ply_file) and os.path.exists(features_file) and os.path.exists(operator_file):
        return None

    vertices, faces = read_vertices_and_triangles(ply_file=ply_file)
    features_dump = np.load(features_file)
    features, confidence = features_dump["features"], features_dump["confidence"]
    vertices, faces, features, confidence = learning_utils.list_from_numpy([vertices, faces, features, confidence])
    frames, mass, _, evals, evecs, grad_x, grad_y = get_operators.surf_to_operators(vertices=vertices,
                                                                                    faces=faces,
                                                                                    npz_path=operator_file)
    return (features, confidence, vertices, mass, torch.rand(1, 3),
            evals, evecs, grad_x.to_dense(), grad_y.to_dense(), faces)


def load_pyg(pyg_dir, name):
    pyg_path = os.path.join(pyg_dir, f"{name}.pth")
    return torch.load(pyg_path)


def dump_pyg(surface, graph, pyg_dir, name, overwrite=False):
    if surface is None or graph is None:
        return
    pyg_path = os.path.join(pyg_dir, f"{name}.pth")
    if os.path.exists(pyg_path) and not overwrite:
        return
    os.makedirs(pyg_dir, exist_ok=True)
    pyg_data = Data(surface=surface, graph=graph)
    torch.save(pyg_data, pyg_path)
