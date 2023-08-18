import os
import sys

import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.io import load_diffnetfiles, load_graph
# from data_processing.main import get_diffnetfiles, get_graph
from data_processing.Atom3DDataset import Atom3DDataset
from data_processing.data_module import SurfaceObject
from data_processing.transforms import Normalizer


class PSRDataset(Atom3DDataset):
    def __init__(self, lmdb_path,
                 geometry_path='../../data/PSR/geometry/',
                 operator_path='../../data/PSR/operator/',
                 graph_path='../../data/PSR/graphs/',
                 big_graphs=False,
                 return_graph=False,
                 return_surface=True,
                 recompute=False,
                 use_xyz=False):
        self.big_graphs = big_graphs
        if big_graphs:
            graph_path = graph_path.replace('graphs', 'big_graphs')
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path,
                         graph_path=graph_path, operator_path=operator_path)
        self.recompute = recompute
        self.return_graph = return_graph
        self.return_surface = return_surface
        self.big_graphs = big_graphs
        self.use_xyz = use_xyz

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
            system = self._lmdb_dataset[index]
            df = system['atoms'].reset_index(drop=True)
            # item[id] has a weird formatting
            name = system['id']
            target, decoy = name[1:-1].split(',')
            target, decoy = target[2:-1], decoy[2:-1]
            name = f"{target}_{decoy}"
            scores = system['scores']

            item = Data(name=name, scores=torch.tensor([scores['gdt_ts']]))
            graph_feat, surface = None, None
            normalizer = Normalizer(add_xyz=self.use_xyz)
            if self.return_surface:
                geom_feats = load_diffnetfiles(name=name,
                                               dump_surf_dir=self.get_geometry_dir(name),
                                               dump_operator_dir=self.get_operator_dir(name))
                if geom_feats is None:
                    raise ValueError("A geometric feature is buggy")

                surface = SurfaceObject(*geom_feats)
                normalizer.set_mean(surface.vertices)
                surface = normalizer.transform_surface(surface)

            if self.return_graph:
                graph_feat = load_graph(name=name, dump_graph_dir=self.get_graph_dir(name))
                if graph_feat is None:
                    raise ValueError("A graph feature is buggy")
                if normalizer.mean is None:
                    normalizer.set_mean(graph_feat.pos)
                graph_feat = normalizer.transform_graph(graph_feat)

            # if both surface and graph are needed, but only one is available, return None to skip the batch
            if (graph_feat is None and self.return_graph) or (surface is None and self.return_surface):
                graph_feat, surface = None, None

            item.graph = graph_feat
            item.surface = surface
            return item

        except Exception as e:
            print("------------------")
            print(f"Error in __getitem__: {e}")
            item = Data()
            return item


if __name__ == '__main__':
    data_dir = '../data/PSR/test/'
    dataset = PSRDataset(data_dir)
    for i, data in enumerate(dataset):
        print(i)
        if i > 5:
            break
