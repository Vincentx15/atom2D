import igl
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

import scipy.spatial as ss
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected

from hmr_min import DataLoaderBase
from hmr_min import res_type_to_hphob
from hmr_min import compute_HKS
from atom2d_utils.learning_utils import list_from_numpy
from data_processing.get_operators import get_operators
from data_processing.data_module import SurfaceObject
from data_processing.data_module import AtomBatch


def atom_coords_to_edges(node_pos, edge_dist_cutoff=4.5):
    r"""
    Turn nodes position into neighbors graph.
    """
    # import time
    # t0 = time.time()
    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()
    edges = to_undirected(edges)
    # print(f"time to pre_dist : {time.time() - t0}")

    # t0 = time.time()
    node_a = node_pos[edges[0, :]]
    node_b = node_pos[edges[1, :]]
    with torch.no_grad():
        my_edge_weights_torch = 1 / (np.linalg.norm(node_a - node_b, axis=1) + 1e-5)
    return edges, my_edge_weights_torch


def load_data_fpaths_from_split_file(data_dir, split_fpath):
    """Load a list of data fpath available under data_dir based on the split list"""
    with open(split_fpath, 'r') as handles:
        data_list = [l.strip('\n').strip() for l in handles.readlines()]
        fpaths = [(kw, list(Path(data_dir).glob(f'{kw}*.npz'))) for kw in data_list]
        not_found = [f[0] for f in fpaths if len(f[1]) == 0]
        fpaths = [f for f_list in fpaths for f in f_list[1]]

        small_patches = {
            # TRAIN
            "2D2I_ACBD_patch_1_NAP.npz",
            "2D2I_ACBD_patch_3_NAP.npz",
            "5K18_ABF_patch_0_COA.npz",
            "4P7A_AB_patch_1_ADP.npz",
            "4FEG_ACBD_patch_3_FAD.npz",
            "1MXB_AB_patch_0_ADP.npz",
            "5U25_AB_patch_1_FAD.npz",
            # VALIDATION
            "1TOX_AC_patch_1_NAD.npz",
            # TEST
            "5C3C_ACBEDF_patch_5_ADP.npz"
        }
        filtered_paths = [f for f in fpaths if f.name not in small_patches]
        if len(not_found) > 0:
            print(f"{len(not_found)} data in the split file not found under data dir")
    return filtered_paths


class DataLoaderMasifLigand(DataLoaderBase):

    def __init__(self, config):
        super().__init__(config)

        self._init_datasets(config)
        self._init_samplers()
        self._init_loaders()

    def _load_split_file(self, split_fpath):
        """Load all matching data (patch) under self.data_dir in the split_fpath"""
        return load_data_fpaths_from_split_file(self.data_dir, split_fpath)

    def _init_datasets(self, config):

        self.data_dir = Path(config.data_dir)
        self.processed_dir = Path(config.processed_dir)

        # load train-valid-test split
        train_fpaths = []
        valid_fpaths = []
        test_fpaths = []

        if config.train_split_file:
            train_fpaths = self._load_split_file(config.train_split_file)
        if config.valid_split_file:
            valid_fpaths = self._load_split_file(config.valid_split_file)
        if config.test_split_file:
            test_fpaths = self._load_split_file(config.test_split_file)

        self.train_set = DatasetMasifLigand(config, train_fpaths)
        self.valid_set = DatasetMasifLigand(config, valid_fpaths)
        self.test_set = DatasetMasifLigand(config, test_fpaths)

        msg = [f'MaSIF-ligand task, train: {len(self.train_set)},',
               f'val: {len(self.valid_set)}, test: {len(self.test_set)}']
        logging.info(' '.join(msg))


class DatasetMasifLigand(Dataset):

    def __init__(self, config, fpaths):
        # feature args
        self.use_chem_feat = config.use_chem_feat
        self.use_geom_feat = config.use_geom_feat
        self.atom_dist = True
        self.atom_angle = True

        self.max_eigen_val = config.max_eigen_val
        self.smoothing = config.smoothing
        self.num_signatures = config.num_signatures

        self.gauss_curv_gdf = GaussianDistance(start=-0.1, stop=0.1, num_centers=config.num_gdf)
        self.mean_curv_gdf = GaussianDistance(start=-0.5, stop=0.5, num_centers=config.num_gdf)
        self.dist_gdf = GaussianDistance(start=0., stop=8., num_centers=config.num_gdf)
        self.angular_gdf = GaussianDistance(start=-1., stop=1., num_centers=config.num_gdf)

        # data dir
        self.data_dir = Path(config.data_dir)
        assert self.data_dir.exists(), f"Dataset dir {self.data_dir} not found"
        self.processed_dir = Path(config.processed_dir)
        self.operator_dir = Path(config.operator_dir)
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.operator_dir.mkdir(exist_ok=True, parents=True)
        self.fpaths = fpaths

    def __getitem__(self, idx):
        """Neighboring atomic environment information is too large to store on HDD,
        we do it on-the-fly for the moment
        """

        # load data
        fpath = self.fpaths[idx]
        fname = Path(fpath).name
        processed_fpath = self.processed_dir / fname
        operator_fpath = self.operator_dir / fname
        success = True
        if not processed_fpath.exists() or not operator_fpath.exists():
            success = preprocess_data(data_fpath=fpath,
                                      processed_fpath=processed_fpath,
                                      operator_fpath=operator_fpath,
                                      max_eigen_val=self.max_eigen_val,
                                      smoothing=self.smoothing,
                                      num_signatures=self.num_signatures)

        if not success:
            # TRAIN
            # 2D2I_ACBD_patch_1_NAP.npz
            # 2D2I_ACBD_patch_3_NAP.npz
            # 5K18_ABF_patch_0_COA.npz
            # 4P7A_AB_patch_1_ADP.npz
            # 4FEG_ACBD_patch_3_FAD.npz
            # 1MXB_AB_patch_0_ADP.npz
            # 5U25_AB_patch_1_FAD.npz

            # VALIDATION
            # 1TOX_AC_patch_1_NAD.npz

            # TEST
            # 5C3C_ACBEDF_patch_5_ADP.npz
            print(fpath)
            return None
        surface_res, graph_res, label = load_preprocessed_data(processed_fpath, operator_fpath)
        label = int(label)
        ##############################  chem feats  ##############################
        # full chemistry features in node_info :
        # res_type  atom_type  hphob  charge  radius  is_alphaC
        # OH 21     OH 12      1      1       1       1

        node_pos, node_info, edge_index, edge_feats = graph_res
        res_hot = np.eye(21, dtype=np.float32)[node_info[:, 0].astype(int)]
        atom_hot = np.eye(12, dtype=np.float32)[node_info[:, 1].astype(int)]
        node_feats = np.concatenate((res_hot, atom_hot, node_info[:, 2:]), axis=1)

        node_pos, node_feats, edge_index, edge_feats = list_from_numpy([node_pos, node_feats, edge_index, edge_feats])
        graph = Data(pos=node_pos, x=node_feats, edge_index=edge_index, edge_feats=edge_feats)

        # GET SURFACE
        mass, L, evals, evecs, grad_x, grad_y, faces, geom_info, verts = surface_res

        ##############################  geom feats  ##############################
        # full geom features
        #     verts   vnormal gauss_curv  mean_curv   signature
        #     0 ~ 2   3 ~ 5   0           1            2 ~ 2 + num_signature - 1

        # expand curvatures to gdf
        gauss_curvs = geom_info[:, 0]
        gauss_curvs_gdf = self.gauss_curv_gdf.expand(gauss_curvs)
        mean_curvs = geom_info[:, 1]
        mean_curvs_gdf = self.mean_curv_gdf.expand(mean_curvs)
        geom_feats = np.concatenate((gauss_curvs_gdf, mean_curvs_gdf, geom_info[:, 2:]), axis=-1)
        geom_feats = torch.from_numpy(geom_feats)
        surface = SurfaceObject(features=geom_feats, confidence=None, vertices=verts, mass=mass, L=L, evals=evals,
                                evecs=evecs, gradX=grad_x, gradY=grad_y, faces=faces, cat_confidence=False)

        item = Data(labels=label, surface=surface, graph=graph)
        return item

    @staticmethod
    def collate_wrapper(unbatched_list):
        unbatched_list = [elt for elt in unbatched_list if elt is not None]
        return AtomBatch.from_data_list(unbatched_list)

    def __len__(self):
        return len(self.fpaths)


def preprocess_data(data_fpath,
                    processed_fpath,
                    operator_fpath,
                    max_eigen_val,
                    smoothing,
                    num_signatures):
    """Preprocess data and cache on disk
    """
    try:

        # load data
        data = np.load(data_fpath, allow_pickle=True)
        label = data['label']

        atom_info = data['atom_info']
        atom_coords = atom_info[:, :3]
        verts = data['pkt_verts']
        faces = data['pkt_faces'].astype(int)

        if max_eigen_val is not None:
            ev = np.where(data['eigen_vals'] < max_eigen_val)[0]
            assert len(ev) > 1
            eigen_vals = data['eigen_vals'][ev]
            eigen_vecs = data['eigen_vecs'][:, ev]
        else:
            eigen_vals = data['eigen_vals']
            eigen_vecs = data['eigen_vecs']
        mass = data['mass'].item()
        eigen_vecs_inv = eigen_vecs.T @ mass

        if smoothing:
            verts = eigen_vecs @ (eigen_vecs_inv @ verts)

        ##############################  atom chem feats  ##############################
        # Atom chemical features
        # x  y  z  res_type  atom_type  charge  radius  is_alphaC
        # 0  1  2  3         4          5       6       7
        # get hphob
        atom_hphob = np.array([[res_type_to_hphob[atom_inf[3]]] for atom_inf in atom_info])
        atom_feats = np.concatenate([atom_info[:, :5], atom_hphob, atom_info[:, 5:]], axis=1)

        # atom_bt = BallTree(atom_coords)
        # vert_nbr_dist, vert_nbr_ind = atom_bt.query(verts, k=vert_nbr_atoms)

        ##############################  Geom feats  ##############################
        vnormals = igl.per_vertex_normals(verts, faces)

        geom_feats = []

        _, _, k1, k2 = igl.principal_curvature(verts, faces)
        gauss_curvs = k1 * k2
        mean_curvs = 0.5 * (k1 + k2)
        geom_feats.extend([gauss_curvs.reshape(-1, 1), mean_curvs.reshape(-1, 1)])
        # HKS:
        geom_feats.append(compute_HKS(eigen_vecs, eigen_vals, num_signatures))

        geom_feats = np.concatenate(geom_feats, axis=-1)
        geom_feats = np.concatenate([verts, vnormals, geom_feats], axis=-1)

        #############################  Laplace-Beltrami basis  ##############################
        # eigs = np.concatenate(
        #     (eigen_vals.reshape(1, -1), eigen_vecs, eigen_vecs_inv.T),
        #     axis=0
        # )

        ##############################  Cache processed  ##############################
        verts = torch.from_numpy(verts)
        faces = torch.from_numpy(faces)
        atom_coords = torch.from_numpy(atom_coords)
        frames, mass, _, evals, evecs, grad_x, grad_y = get_operators(verts=verts,
                                                                      faces=faces,
                                                                      npz_path=operator_fpath)
        edge_index, edge_feats = atom_coords_to_edges(node_pos=atom_coords)
        np.savez(processed_fpath,
                 label=label.astype(np.int8),
                 # input
                 node_pos=atom_coords,
                 node_info=atom_feats[:, 3:].astype(np.float32),
                 edge_index=edge_index,
                 edge_feats=edge_feats,
                 verts=verts,
                 faces=faces,
                 geom_info=geom_feats.astype(np.float32),)
        return True
    except:
        return False
    # np.savez(
    #     out_fpath,
    #     label=label.astype(np.int8),
    #     # input
    #     atom_info=atom_feats.astype(np.float32),
    #     geom_info=geom_feats.astype(np.float32),
    #     eigs=eigs.astype(np.float32),
    #     # vert_nbr
    #     vert_nbr_dist=vert_nbr_dist.astype(np.float32),
    #     vert_nbr_ind=vert_nbr_ind.astype(np.int32)
    # )


def load_preprocessed_data(processed_fpath, operator_path):
    data = np.load(processed_fpath, allow_pickle=True)
    label = data['label']

    # GET GRAPH
    node_pos = data['node_pos']
    node_info = data['node_info']
    edge_index = data['edge_index']
    edge_feats = data['edge_feats']

    ##############################  chem feats  ##############################
    # full chemistry features in node_info :
    # res_type  atom_type  hphob  charge  radius  is_alphaC
    # OH 21     OH 11      1      1       1       1

    graph_res = node_pos, node_info, edge_index, edge_feats

    # GET SURFACE
    geom_info = data['geom_info']
    verts = data['verts']
    faces = data['faces']
    verts = torch.from_numpy(verts)
    faces = torch.from_numpy(faces)
    frames, mass, _, evals, evecs, grad_x, grad_y = get_operators(verts=verts,
                                                                  faces=faces,
                                                                  npz_path=operator_path)
    grad_x = SparseTensor.from_torch_sparse_coo_tensor(grad_x.float())
    grad_y = SparseTensor.from_torch_sparse_coo_tensor(grad_y.float())
    surface_res = mass, torch.rand(1, 3), evals, evecs, grad_x, grad_y, faces, geom_info, verts

    return surface_res, graph_res, label


class GaussianDistance(object):
    def __init__(self, start, stop, num_centers):
        self.filters = np.linspace(start, stop, num_centers, dtype=np.float32)
        self.var = (stop - start) / (num_centers - 1)

    def expand(self, d):
        return np.exp(-0.5 * (d[..., None] - self.filters) ** 2 / self.var ** 2)
