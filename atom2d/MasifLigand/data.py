import os
import igl
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

import scipy.spatial as ss
from sklearn.neighbors import BallTree
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from data_processing.hmr_min import DataLoaderBase
from data_processing.hmr_min import res_type_to_hphob
from data_processing.hmr_min import compute_HKS, atom_coords_to_edges
from atom2d_utils.learning_utils import list_from_numpy
from data_processing.get_operators import get_operators
from data_processing.data_module import SurfaceObject
from data_processing.data_module import AtomBatch
from data_processing.add_seq_embs import compute_esm_embs, get_esm_embs


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

        atom_bt = BallTree(atom_coords)
        vert_nbr_atoms = 16
        vert_nbr_dist, vert_nbr_ind = atom_bt.query(verts, k=vert_nbr_atoms)

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

        ##############################  Cache processed  ##############################
        verts = torch.from_numpy(verts)
        faces = torch.from_numpy(faces)
        atom_coords = torch.from_numpy(atom_coords)
        frames, mass, _, evals, evecs, grad_x, grad_y = get_operators(verts=verts,
                                                                      faces=faces,
                                                                      k_eig=eigen_vecs.shape[1],
                                                                      npz_path=operator_fpath)
        edge_index, edge_feats = atom_coords_to_edges(node_pos=atom_coords)
        np.savez(processed_fpath,
                 label=label.astype(np.int8),
                 neig=eigen_vecs.shape[1],
                 # input
                 node_pos=atom_coords,
                 node_info=atom_feats[:, 3:].astype(np.float32),
                 atom_feats=atom_feats.astype(np.float32),
                 edge_index=edge_index,
                 edge_feats=edge_feats,
                 verts=verts,
                 faces=faces,
                 geom_info=geom_feats.astype(np.float32),
                 vert_nbr_dist=vert_nbr_dist.astype(np.float32),
                 vert_nbr_ind=vert_nbr_ind.astype(np.int32), )
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


def load_preprocessed_data(processed_fpath, operator_path, self=None):
    data = np.load(processed_fpath, allow_pickle=True)
    label = data['label']

    # GET GRAPH
    node_pos = data['node_pos']
    node_info = data['node_info']
    edge_index = data['edge_index']
    edge_attr = data['edge_feats']

    ##############################  chem feats  ##############################
    # full chemistry features in node_info :
    # res_type  atom_type  hphob  charge  radius  is_alphaC
    # OH 21     OH 11      1      1       1       1

    graph_res = node_pos, node_info, edge_index, edge_attr

    # GET SURFACE
    geom_info = data['geom_info']
    verts = data['verts']
    faces = data['faces']
    verts = torch.from_numpy(verts)
    faces = torch.from_numpy(faces)
    frames, mass, _, evals, evecs, grad_x, grad_y = get_operators(verts=verts,
                                                                  faces=faces,
                                                                  k_eig=data['neig'],
                                                                  npz_path=operator_path)
    grad_x = SparseTensor.from_torch_sparse_coo_tensor(grad_x.float())
    grad_y = SparseTensor.from_torch_sparse_coo_tensor(grad_y.float())
    surface_res = mass, torch.rand(1, 3), evals, evecs, grad_x, grad_y, faces, geom_info, verts

    # HMR chemical features
    atom_coords = data["atom_feats"][:, :3]
    atom_feats = data["atom_feats"][:, 3:]
    verts = data["geom_info"][:, :3]
    vnormals = data["geom_info"][:, 3:6]
    vert_nbr_dist = data["vert_nbr_dist"]
    vert_nbr_ind = data["vert_nbr_ind"]
    chem_feats = atom_feats[:, [2, 3]]
    dist_flat = np.concatenate(vert_nbr_dist, axis=0)
    ind_flat = np.concatenate(vert_nbr_ind, axis=0)
    # vert-to-atom mapper
    nbr_vid = np.concatenate([[i] * len(vert_nbr_ind[i]) for i in range(len(vert_nbr_ind))])
    chem_feats = [chem_feats[ind_flat]]
    # atom_dist
    chem_feats.append(self.dist_gdf.expand(dist_flat))
    # atom angular
    nbr_vec = atom_coords[ind_flat] - verts[nbr_vid]
    nbr_vnormals = vnormals[nbr_vid]
    nbr_angular = np.einsum("vj,vj->v", nbr_vec / np.linalg.norm(nbr_vec, axis=-1, keepdims=True), nbr_vnormals)
    nbr_angular_gdf = self.angular_gdf.expand(nbr_angular)
    chem_feats.append(nbr_angular_gdf)
    chem_feats = np.concatenate(chem_feats, axis=-1)
    chem_feats = torch.from_numpy(chem_feats).float()

    # HMR geometric features
    geom_feats_in = data["geom_info"][:, 6:]
    gauss_curvs = geom_feats_in[:, 0]
    gauss_curvs_gdf = self.gauss_curv_gdf.expand(gauss_curvs)
    mean_curvs = geom_feats_in[:, 1]
    mean_curvs_gdf = self.mean_curv_gdf.expand(mean_curvs)
    geom_feats = np.concatenate((gauss_curvs_gdf, mean_curvs_gdf, geom_feats_in[:, 2:]), axis=-1)
    geom_feats = torch.from_numpy(geom_feats).float()

    # misc
    nbr_vids = torch.tensor(nbr_vid, dtype=torch.int64)

    return surface_res, graph_res, label, chem_feats, geom_feats, nbr_vids


class GaussianDistance(object):
    def __init__(self, start, stop, num_centers):
        self.filters = np.linspace(start, stop, num_centers, dtype=np.float32)
        self.var = (stop - start) / (num_centers - 1)

    def expand(self, d):
        return np.exp(-0.5 * (d[..., None] - self.filters) ** 2 / self.var ** 2)


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

    def __init__(self, config, use_pronet=False):
        super().__init__(config)

        self._init_datasets(config, use_pronet)
        self._init_samplers()
        self._init_loaders()

    def _load_split_file(self, split_fpath):
        """Load all matching data (patch) under self.data_dir in the split_fpath"""
        return load_data_fpaths_from_split_file(self.data_dir, split_fpath)

    def _init_datasets(self, config, use_pronet=False):

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

        if use_pronet:
            self.train_set = DatasetMasifLigandPronet(config, train_fpaths)
            self.valid_set = DatasetMasifLigandPronet(config, valid_fpaths)
            self.test_set = DatasetMasifLigandPronet(config, test_fpaths)

        else:
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
        self.skip_hydro = config.skip_hydro

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
        self.pdb_dir = self.data_dir.parent / 'raw_data_MasifLigand/pdb'
        self.seq_emb_dir = self.data_dir.parent / 'computed_embs'
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.operator_dir.mkdir(exist_ok=True, parents=True)
        self.seq_emb_dir.mkdir(exist_ok=True, parents=True)
        self.fpaths = fpaths
        self.use_graph_only = config.use_graph_only
        self.add_seq_emb = config.add_seq_emb
        self.recompute_surf = False

    @staticmethod
    def collate_wrapper(unbatched_list):
        unbatched_list = [elt for elt in unbatched_list if elt is not None]
        return AtomBatch.from_data_list(unbatched_list)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        """Neighboring atomic environment information is too large to store on HDD,
        we do it on-the-fly for the moment
        """

        # load data
        fpath = self.fpaths[idx]
        fname = Path(fpath).name
        pdb_code, chains = fname.split('_')[:2]
        pdb_chains = '_'.join((pdb_code, chains))
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
        surface_res, graph_res, label, chem_feats, geom_feats_, nbr_vids = load_preprocessed_data(processed_fpath,
                                                                                                  operator_fpath, self)

        if self.add_seq_emb and self.recompute_surf:
            compute_esm_embs(pdb=pdb_chains + '.pdb',
                             pdb_dir=self.pdb_dir,
                             out_emb_dir=self.seq_emb_dir,
                             recompute=False)

        label = int(label)
        ##############################  chem feats  ##############################
        # full chemistry features in node_info :
        # res_type  atom_type  hphob  charge  radius  is_alphaC
        # OH 21     OH 12      1      1       1       1

        node_pos, node_info, edge_index, edge_attr = graph_res
        res_hot = np.eye(21, dtype=np.float32)[node_info[:, 0].astype(int)]
        atom_hot = np.eye(12, dtype=np.float32)[node_info[:, 1].astype(int)]
        node_feats = np.concatenate((res_hot, atom_hot, node_info[:, 2:]), axis=1)
        node_pos, node_feats, edge_index, edge_attr = list_from_numpy([node_pos, node_feats, edge_index, edge_attr])

        # node_feat[-1] is CA position, its almost residue id with offset of one because of nitrogen.
        ca_loc = node_feats[:, -1]
        offset = torch.cat((ca_loc[1:], torch.zeros(1)))
        atom_to_res_map = torch.cumsum(offset, dim=0)

        # Now concatenate the embeddings
        if self.add_seq_emb:
            esm_embs = get_esm_embs(pdb=pdb_chains, out_emb_dir=self.seq_emb_dir)
            if esm_embs is None:
                print('Failed to load embs', pdb_chains)
                return None
            if atom_to_res_map.max().item() > len(esm_embs):
                print('Max # res is longer than embeddings', pdb_chains)
                return None
            esm_embs = torch.from_numpy(esm_embs)
            expanded_esm_embs = esm_embs[atom_to_res_map.long() - 1]
            node_feats = torch.concatenate((node_feats, expanded_esm_embs), axis=-1)

        graph = Data(pos=node_pos, x=node_feats, edge_index=edge_index, edge_attr=edge_attr,
                     atom_to_res_map=atom_to_res_map)
        if self.skip_hydro:
            not_hydro = np.where(node_info[:, 1] > 0)[0]
            graph = graph.subgraph(torch.from_numpy(not_hydro))

        # GET SURFACE
        if self.use_graph_only:
            surface = None
        else:
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
                                    evecs=evecs, gradX=grad_x, gradY=grad_y, faces=faces, cat_confidence=False,
                                    chem_feats=chem_feats, geom_feats=geom_feats_, nbr_vids=nbr_vids)

        item = Data(labels=label, surface=surface, graph=graph)
        return item


class DatasetMasifLigandPronet(Dataset):

    def __init__(self, config, fpaths):
        # data dir
        self.data_dir = Path(config.data_dir)
        assert self.data_dir.exists(), f"Dataset dir {self.data_dir} not found"
        self.processed_dir = Path(config.processed_dir)
        self.pdb_dir = self.data_dir.parent / 'raw_data_MasifLigand/pdb'
        self.seq_emb_dir = self.data_dir.parent / 'computed_embs'
        self.pronet_dir = self.data_dir.parent / 'pronet'
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        self.seq_emb_dir.mkdir(exist_ok=True, parents=True)
        self.fpaths = fpaths
        self.add_seq_emb = config.add_seq_emb

    @staticmethod
    def collate_wrapper(unbatched_list):
        unbatched_list = [elt for elt in unbatched_list if elt is not None]
        return AtomBatch.from_data_list(unbatched_list)

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        """
        """
        # load data
        fpath = self.fpaths[idx]
        fname = Path(fpath).name
        pdb_code, chains = fname.split('_')[:2]
        pdb_chains = '_'.join((pdb_code, chains))
        processed_fpath = self.processed_dir / fname

        # GET PRONET GRAPH
        pronet_path = os.path.join(self.pronet_dir, pdb_chains + "pronetgraph.pt")
        pronet_graph = torch.load(pronet_path)

        if pronet_graph.coords_ca.isnan().any():
            print('missing CA')
            return None

        if (pronet_graph.coords_n.isnan().any() or pronet_graph.coords_c.isnan().any()
                or pronet_graph.bb_embs.isnan().any()
                or pronet_graph.x.isnan().any()
                or pronet_graph.side_chain_embs.isnan().any()):
            print('missing something')
            return None

        # Now concatenate the embeddings
        if self.add_seq_emb:
            esm_embs = get_esm_embs(pdb=pdb_chains, out_emb_dir=self.seq_emb_dir)
            if esm_embs is None:
                print('Failed to load embs', pdb_chains)
                return None
            if len(pronet_graph.coords_ca) != len(esm_embs):
                print('Max # res is longer than embeddings', pdb_chains)
                return None
            esm_embs = torch.from_numpy(esm_embs)
            pronet_graph.seq_emb = esm_embs
        data = np.load(processed_fpath, allow_pickle=True)
        label = data['label']
        label = int(label)
        verts = data['verts']
        verts = torch.from_numpy(verts)

        item = Data(labels=label, verts=verts, pronet_graph=pronet_graph)
        return item
