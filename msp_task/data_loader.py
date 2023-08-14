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
from data_processing.transforms import AddMSPTransform
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
            coords = orig_coords, mut_coords
            coords = list_from_numpy(coords)
            coords = [x.float() for x in coords]

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

                surface_lo = SurfaceObject(*geom_feats[0], coords=coords[0]) if geom_feats[0] is not None else None
                surface_ro = SurfaceObject(*geom_feats[1]) if geom_feats[1] is not None else None
                all_verts = [surface_lo.vertices, surface_ro.vertices]  # for the transform
                surface_lo.all_vertices, surface_ro.all_vertices = [x.clone() for x in all_verts], [x.clone() for x in all_verts]
                surface_lo = self.transform(surface_lo)
                surface_ro = self.transform(surface_ro)

                surface_lm = SurfaceObject(*geom_feats[2], coords=coords[1]) if geom_feats[2] is not None else None
                surface_rm = SurfaceObject(*geom_feats[3]) if geom_feats[3] is not None else None
                all_verts = [surface_lm.vertices, surface_rm.vertices]  # for the transform
                surface_lm.all_vertices, surface_rm.all_vertices = [x.clone() for x in all_verts], [x.clone() for x in all_verts]
                surface_lm = self.transform(surface_lm)
                surface_rm = self.transform(surface_rm)

                if surface_lo is None or surface_ro is None or surface_lm is None or surface_rm is None:
                    surface_lo, surface_ro, surface_lm, surface_rm = None, None, None, None
                    raise ValueError("A geometric feature is buggy")

            if self.return_graph:
                graph_feats = [get_graph(name=name, df=df,
                                         dump_graph_dir=self.get_graph_dir(name),
                                         big=self.big_graphs,
                                         recompute=True)
                               for i, (name, df) in enumerate(zip(names, dfs))]
                graph_lo, graph_ro, graph_lm, graph_rm = graph_feats[0], graph_feats[1], graph_feats[2], graph_feats[3]
                if graph_lo is None or graph_ro is None or graph_lm is None or graph_rm is None:
                    graph_lo, graph_ro, graph_lm, graph_rm = None, None, None, None
                    raise ValueError("A graph feature is buggy")

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
