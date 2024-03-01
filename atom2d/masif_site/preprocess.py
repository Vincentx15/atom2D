import os
import sys

import numpy as np
import pymesh
import torch
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

if __name__ == '__main__':
    sys.path.append('..')

from data_processing.hmr_min import compute_HKS
from data_processing.hmr_min import res_type_to_hphob, pdb_to_atom_info, atom_coords_to_edges
from data_processing.get_operators import get_operators


def normalize_electrostatics(in_elec):
    """
        Normalize electrostatics to a value between -1 and 1
    """
    elec = np.copy(in_elec)
    upper_threshold = 3
    lower_threshold = -3
    elec[elec > upper_threshold] = upper_threshold
    elec[elec < lower_threshold] = lower_threshold
    elec = elec - lower_threshold
    elec = elec / (upper_threshold - lower_threshold)
    elec = 2 * elec - 1
    return elec


def preprocess_one(pdb_name,
                   ply_dir='../../data/masif_site/01-benchmark_surfaces/',
                   pdb_dir='../../data/masif_site/01-benchmark_pdbs/',
                   processed_dir='../../data/masif_site/processed'):
    """
   Read data from a ply file -- decompose into patches.
   Returns:
   list_desc: List of features per patch
   list_coords: list of angular and polar coordinates.
   list_indices: list of indices of neighbors in the patch.
   list_sc_labels: list of shape complementarity labels (computed here).
   """
    pdb = os.path.join(pdb_dir, pdb_name + '.pdb')
    ply = os.path.join(ply_dir, pdb_name + '.ply')
    operator_path = os.path.join(processed_dir, pdb_name + '_operator.npz')
    processed_path = os.path.join(processed_dir, pdb_name + '_processed.npz')
    # if os.path.exists(processed_path):
    #     return 1

    mesh = pymesh.load_mesh(ply)

    # Get operator
    verts = torch.from_numpy(np.copy(mesh.vertices))
    faces = torch.from_numpy(np.copy(mesh.faces))
    frames, mass, _, evals, evecs, grad_x, grad_y = get_operators(verts=verts,
                                                                  faces=faces,
                                                                  npz_path=operator_path)
    # Normals:
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1, n2, n3], axis=1)

    # Compute the principal curvature components for the shape index.
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)

    # In addition to those original features, we add the HKS following HMR processing
    num_signatures = 16
    hks = compute_HKS(eigen_vecs=evecs.numpy(), eigen_vals=evals.numpy(), num_t=num_signatures)
    # shape (N,1+1+1+16+3)
    geom_feats = np.concatenate((H[:, None], K[:, None], si[:, None], hks, normals), axis=1)

    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob") / 4.5
    # shape (N,1+1+1)
    chem_feats = np.concatenate((charge[:, None], hbond[:, None], hphob[:, None]), axis=1)
    surface_feats = np.concatenate((geom_feats, chem_feats), axis=1)

    ##############################  atom chem feats  ##############################
    # Atom chemical features
    # x  y  z  res_type  atom_type  charge  radius  is_alphaC
    # 0  1  2  3         4          5       6       7
    # get hphob
    atom_info = pdb_to_atom_info(pdb)
    atom_coords = torch.from_numpy(atom_info[:, :3])
    atom_hphob = np.array([[res_type_to_hphob[atom_inf[3]]] for atom_inf in atom_info])
    atom_feats = np.concatenate([atom_info[:, :5], atom_hphob, atom_info[:, 5:]], axis=1)
    edge_index, edge_feats = atom_coords_to_edges(node_pos=atom_coords)

    # Iface labels (for ground truth only)
    iface_labels = mesh.get_attribute("vertex_iface")

    np.savez(processed_path,
             label=iface_labels,
             node_pos=atom_coords,
             node_feats=atom_feats[:, 3:].astype(np.float32),
             edge_index=edge_index,
             edge_feats=edge_feats,
             verts=verts,
             faces=faces,
             surface_feats=surface_feats.astype(np.float32), )
    return 1


class PreProcessorMasifSite(Dataset):

    def __init__(self,
                 train_list='../../data/masif_site/train_list.txt',
                 test_list='../../data/masif_site/test_list.txt',
                 ply_dir='../../data/masif_site/01-benchmark_surfaces/',
                 pdb_dir='../../data/masif_site/01-benchmark_pdbs/',
                 processed_dir='../../data/masif_site/processed'):
        train_names = set([name.strip() for name in open(train_list, 'r').readlines()])
        test_names = set([name.strip() for name in open(test_list, 'r').readlines()])
        self.all_sys = list(train_names.union(test_names))

        # feature args
        self.ply_dir = ply_dir
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    def __len__(self):
        return len(self.all_sys)

    def __getitem__(self, idx):
        pdb_name = self.all_sys[idx]
        try:
            success = preprocess_one(pdb_name=pdb_name,
                                     ply_dir=self.ply_dir,
                                     pdb_dir=self.pdb_dir,
                                     processed_dir=self.processed_dir)
        except Exception as e:
            print(e)
            success = 0
        return success


def do_all():
    dataset = PreProcessorMasifSite()
    dataloader = DataLoader(dataset, num_workers=4, batch_size=1)
    total_success = 0
    for i, success in enumerate(dataloader):
        total_success += int(success)
        if not i % 50:
            print(f'Processed {i + 1}/{len(dataloader)}, with {total_success} successes')
            # => Processed 3351/3362, with 3328 successes ~1% failed systems, mostly missing ply files


if __name__ == '__main__':
    # preprocess_one('3EEY_B')
    do_all()
