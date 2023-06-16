import os
import sys

import torch

from atom3d.util.formats import get_coordinates_from_df

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing import main
from data_processing.preprocessor_dataset import Atom3DDataset
from atom2d_utils.learning_utils import list_from_numpy


class MSPDataset(Atom3DDataset):
    def __init__(self, lmdb_path,
                 geometry_path='../../data/MSP/geometry/',
                 operator_path='../../data/MSP/operator/',
                 graph_path='../../data/MSP/graph',
                 return_graph=False,
                 recompute=False):
        super().__init__(lmdb_path=lmdb_path, geometry_path=geometry_path,
                         operator_path=operator_path, graph_path=graph_path)
        self.recompute = recompute
        self.return_graph = return_graph

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

            # Then get the split dfs and names, and retrieve the surfaces
            # Apparently this is faster than split
            left_orig = orig_df[orig_df['chain'].isin(list(chains_left))]
            right_orig = orig_df[orig_df['chain'].isin(list(chains_right))]
            left_mut = mut_df[mut_df['chain'].isin(list(chains_left))]
            right_mut = mut_df[mut_df['chain'].isin(list(chains_right))]

            names = [f"{pdb}_{chains_left}", f"{pdb}_{chains_right}",
                     f"{pdb}_{chains_left}_{mutation}", f"{pdb}_{chains_right}_{mutation}"]
            dfs = [left_orig, right_orig, left_mut, right_mut]

            geom_feats = [main.get_diffnetfiles(name=name, df=df,
                                                dump_surf_dir=self.get_geometry_dir(name),
                                                dump_operator_dir=self.get_operator_dir(name),
                                                recompute=self.recompute)
                          for name, df in zip(names, dfs)]
            if any([geom_feat is None for geom_feat in geom_feats]):
                raise ValueError("A geometric feature is buggy")

            if self.return_graph:
                graph_feats = [main.get_graph(name=name, df=df,
                                              dump_graph_dir=self.get_graph_dir(name),
                                              recompute=True)  # TODO : fix
                               for name, df in zip(names, dfs)]
                if any([graph_feat is None for graph_feat in graph_feats]):
                    raise ValueError("A graph feature is buggy")
                return names, geom_feats, coords, torch.tensor([float(item['label'])]), graph_feats

            return names, geom_feats, coords, torch.tensor([float(item['label'])])
        except Exception as e:
            print("------------------")
            print(f"Error in __getitem__: {e}")
            if self.return_graph:
                return None, None, None, None, None
            else:
                return None, None, None, None


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
