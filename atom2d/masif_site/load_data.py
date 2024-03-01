import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_sparse import SparseTensor

if __name__ == '__main__':
    sys.path.append('..')

from atom2d_utils.learning_utils import list_from_numpy
from data_processing.get_operators import get_operators
from data_processing.data_module import SurfaceObject
from data_processing.data_module import AtomBatch


def load_preprocessed_data(processed_fpath, operator_path):
    """
    Almost identical to MasifLigand/data.py but name for geom feats is different
    :param processed_fpath:
    :param operator_path:
    :return:
    """
    data = np.load(processed_fpath, allow_pickle=True)

    # GET GRAPH
    node_pos = data['node_pos']
    node_info = data['node_feats']
    edge_index = data['edge_index']
    edge_attr = data['edge_feats']
    graph_res = node_pos, node_info, edge_index, edge_attr

    # GET SURFACE
    geom_feats = data['surface_feats']
    verts = data['verts']
    faces = data['faces']
    verts = torch.from_numpy(verts)
    faces = torch.from_numpy(faces)
    frames, mass, _, evals, evecs, grad_x, grad_y = get_operators(verts=verts,
                                                                  faces=faces,
                                                                  npz_path=operator_path)
    grad_x = SparseTensor.from_torch_sparse_coo_tensor(grad_x.float())
    grad_y = SparseTensor.from_torch_sparse_coo_tensor(grad_y.float())
    surface_res = mass, torch.rand(1, 3), evals, evecs, grad_x, grad_y, faces, geom_feats, verts

    labels = data['label']
    return surface_res, graph_res, labels


class DatasetMasifSite(Dataset):

    def __init__(self,
                 systems_list='../../data/masif_site/train_list.txt',
                 processed_dir='../../data/masif_site/processed'):
        all_sys = [name.strip() for name in open(systems_list, 'r').readlines()]
        self.processed_dir = processed_dir
        successful_operators_pdb = [file.rstrip('_operator.npz') for file in os.listdir(self.processed_dir)]
        successful_processed_pdb = [file.rstrip('_processed.npz') for file in os.listdir(self.processed_dir)]
        self.all_sys = list(set(all_sys).intersection(successful_operators_pdb).intersection(successful_processed_pdb))
        print(len(all_sys))
        print(len(self.all_sys))
        self.skip_hydro = False

    def __len__(self):
        return len(self.all_sys)

    def __getitem__(self, idx):
        pdb_name = self.all_sys[idx]
        operator_path = os.path.join(self.processed_dir, pdb_name + '_operator.npz')
        processed_path = os.path.join(self.processed_dir, pdb_name + '_processed.npz')
        surface_res, graph_res, labels = load_preprocessed_data(processed_path, operator_path)

        # GRAPH CONSTRUCTION
        ##############################  chem feats  ##############################
        # full chemistry features in node_info :
        # res_type  atom_type  hphob  charge  radius  is_alphaC
        # OH 21     OH 11      1      1       1       1

        node_pos, node_info, edge_index, edge_attr = graph_res
        res_hot = np.eye(21, dtype=np.float32)[node_info[:, 0].astype(int)]
        atom_hot = np.eye(12, dtype=np.float32)[node_info[:, 1].astype(int)]
        node_feats = np.concatenate((res_hot, atom_hot, node_info[:, 2:]), axis=1)
        node_pos, node_feats, edge_index, edge_attr = list_from_numpy([node_pos, node_feats, edge_index, edge_attr])
        graph = Data(pos=node_pos, x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
        if self.skip_hydro:
            not_hydro = np.where(node_info[:, 1] > 0)[0]
            graph = graph.subgraph(torch.from_numpy(not_hydro))

        # SURFACE CONSTRUCTION
        mass, L, evals, evecs, grad_x, grad_y, faces, geom_feats, verts = surface_res
        surface = SurfaceObject(features=geom_feats, confidence=None, vertices=verts, mass=mass, L=L, evals=evals,
                                evecs=evecs, gradX=grad_x, gradY=grad_y, faces=faces, cat_confidence=False)
        labels = torch.from_numpy(labels)
        item = Data(labels=labels, surface=surface, graph=graph)
        return item


if __name__ == '__main__':
    from torch_geometric.data import DataLoader

    dataset = DatasetMasifSite()
    dataloader = DataLoader(dataset, num_workers=4, batch_size=1)
    for i, item in enumerate(dataloader):
        print(i, item)
        if not i % 10:
            print(i)