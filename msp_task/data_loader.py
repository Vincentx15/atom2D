import os
import sys

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from atom3d.util.formats import get_coordinates_from_df

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.main import get_diffnetfiles, get_graph
from data_processing.preprocessor_dataset import Atom3DDataset
from data_processing.data_module import SurfaceObject
from data_processing.transforms import AddMSPTransform, Normalizer
from atom2d_utils.learning_utils import list_from_numpy


class MSPDataset(Atom3DDataset):
    def __init__(self, lmdb_path,
                 geometry_path='../../data/MSP/geometry/',
                 operator_path='../../data/MSP/operator/',
                 graph_path='../../data/MSP/graph',
                 return_graph=False,
                 big_graphs=False,
                 return_surface=True,
                 recompute=False,
                 use_xyz=False):
        self.big_graphs = big_graphs
        if big_graphs:
            graph_path = graph_path.replace('graphs', 'big_graphs')
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path,
                         operator_path=operator_path, graph_path=graph_path)
        self.recompute = recompute
        self.return_graph = return_graph
        self.return_surface = return_surface
        self.use_xyz = use_xyz

        transforms = [AddMSPTransform(use_xyz)]
        self.transform = T.Compose(transforms)

    @staticmethod
    def _extract_mut_idx(df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        return torch.LongTensor(idx)

    def __getitem__(self, index):
        """

        :param index:
        :return: pos and neg arrays of the 2 partners CA 3D coordinates shape N_{pos,neg}x 2x 3
                 and the geometry objects necessary to embed the surfaces
        """

        try:
            item = self._lmdb_dataset[index]

            # mutation is like AD56G which means Alanine (A) in chain D residue number 56 (D56) -> Glycine (G)
            pdb, chains_left, chains_right, mutation = item['id'].split('_')

            # First get ids in the databases and their coords
            orig_df = item['original_atoms'].reset_index(drop=True)
            mut_df = item['mutated_atoms'].reset_index(drop=True)
            orig_idx = self._extract_mut_idx(orig_df, mutation)
            mut_idx = self._extract_mut_idx(mut_df, mutation)
            orig_coords = get_coordinates_from_df(orig_df.iloc[orig_idx])
            mut_coords = get_coordinates_from_df(mut_df.iloc[mut_idx])
            orig_coords = torch.from_numpy(orig_coords).float()
            mut_coords = torch.from_numpy(mut_coords).float()

            normalizer_orig = Normalizer(add_xyz=self.use_xyz).set_mean(orig_coords)
            normalizer_mut = Normalizer(add_xyz=self.use_xyz).set_mean(mut_coords)
            orig_coords = normalizer_orig.transform(orig_coords)
            mut_coords = normalizer_mut.transform(mut_coords)
            coords = [orig_coords, mut_coords]


            names = [f"{pdb}_{chains_left}", f"{pdb}_{chains_right}",
                     f"{pdb}_{chains_left}_{mutation}", f"{pdb}_{chains_right}_{mutation}"]
            batch = Data(names=names, coords=coords, label=torch.tensor([float(item['label'])]))

            # Then get the split dfs and names, and retrieve the surfaces
            # Apparently this is faster than split
            left_orig = orig_df[orig_df['chain'].isin(list(chains_left))]
            right_orig = orig_df[orig_df['chain'].isin(list(chains_right))]
            left_mut = mut_df[mut_df['chain'].isin(list(chains_left))]
            right_mut = mut_df[mut_df['chain'].isin(list(chains_right))]
            dfs = [left_orig, right_orig, left_mut, right_mut]

            graph_lo, graph_ro, graph_lm, graph_rm = None, None, None, None
            surface_lo, surface_ro, surface_lm, surface_rm = None, None, None, None

            if self.return_surface:
                geom_feats = [get_diffnetfiles(name=name,
                                               df=df,
                                               dump_surf_dir=self.get_geometry_dir(name),
                                               dump_operator_dir=self.get_operator_dir(name),
                                               recompute=self.recompute)
                              for name, df in zip(names, dfs)]
                if any([geom is None for geom in geom_feats]):
                    raise ValueError("A geometric feature is buggy")
                else:
                    surface_lo, surface_ro, surface_lm, surface_rm = [SurfaceObject(*geom_feat) for geom_feat in geom_feats]
                    surface_lo = normalizer_orig.transform_surface(surface_lo)
                    surface_ro = normalizer_orig.transform_surface(surface_ro)
                    surface_lm = normalizer_mut.transform_surface(surface_lm)
                    surface_rm = normalizer_mut.transform_surface(surface_rm)

            if self.return_graph:
                graph_feats = [get_graph(name=name, df=df,
                                         dump_graph_dir=self.get_graph_dir(name),
                                         big=self.big_graphs,
                                         recompute=True)
                               for i, (name, df) in enumerate(zip(names, dfs))]
                if any([graph is None for graph in graph_feats]):
                    raise ValueError("A graph feature is buggy")
                else:
                    graph_lo, graph_ro, graph_lm, graph_rm = graph_feats
                    graph_lo = normalizer_orig.transform_graph(graph_lo)
                    graph_ro = normalizer_orig.transform_graph(graph_ro)
                    graph_lm = normalizer_mut.transform_graph(graph_lm)
                    graph_rm = normalizer_mut.transform_graph(graph_rm)


            # if both surface and graph are needed, but only one is available, return None to skip the batch
            if (graph_lo is None and self.return_graph) or (surface_lo is None and self.return_surface):
                graph_lo, graph_ro, graph_lm, graph_rm = None, None, None, None
                surface_lo, surface_ro, surface_lm, surface_rm = None, None, None, None

            batch.surface_lo = surface_lo
            batch.surface_ro = surface_ro
            batch.surface_lm = surface_lm
            batch.surface_rm = surface_rm
            batch.graph_lo = graph_lo
            batch.graph_ro = graph_ro
            batch.graph_lm = graph_lm
            batch.graph_rm = graph_rm
            return batch

        except Exception as e:
            print("------------------")
            print(f"Error in __getitem__: {e}")
            batch = Data()
            return batch


if __name__ == '__main__':
    data_dir = '../data/MSP/test/'
    dataset = MSPDataset(data_dir,
                         geometry_path='../data/MSP/geometry/',
                         operator_path='../data/MSP/operator/',
                         graph_path='../data/MSP/graph',
                         return_graph=True)
    for i, data in enumerate(dataset):
        print(i)
        if i > 5:
            break
