import os
import sys

import math
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom2d_utils import naming_utils
from data_processing import main


class NewPIP(torch.utils.data.Dataset):
    def __init__(self, data_dir, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=5,
                 geometry_path='../../data/processed_data/geometry/',
                 operator_path='../../data/processed_data/operator/',
                 graph_path='../../data/processed_data/graph',
                 big_graphs=False,
                 return_graph=False,
                 return_surface=True,
                 recompute=False):
        self.big_graphs = big_graphs
        if big_graphs:
            graph_path = graph_path.replace('graphs', 'big_graphs')
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.recompute = recompute
        csv_path = os.path.join(data_dir, 'all_systems.csv')
        self.dir_path = os.path.join(data_dir, 'systems')
        self.df = pd.read_csv(csv_path)
        self.geometry_path = geometry_path
        self.operator_path = operator_path
        self.graph_path = graph_path

        self.return_surface = return_surface
        self.return_graph = return_graph

    def get_geometry_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.geometry_path)

    def get_operator_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.operator_path)

    def get_graph_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.graph_path)

    def _num_to_use(self, num_pos, num_neg):
        """
        Depending on the number of pos and neg of the system, we might want to use
            different amounts of positive or negative coordinates.

        :param num_pos:
        :param num_neg:
        :return:
        """

        if self.neg_to_pos_ratio == -1:
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
            if self.max_pos_regions_per_ensemble != -1:
                num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    def __len__(self):
        return len(self.df)

    def get_item(self, index):
        # if not index == 42:
        #     return None
        row = self.df.iloc[index][["system", "name1", "name2"]]
        dirname, name1, name2 = row.values
        dirpath = naming_utils.name_to_dir(dirname, dir_path=self.dir_path)
        dump_path = os.path.join(dirpath, dirname)
        dump_struct_1 = os.path.join(dump_path, "struct_1.csv")
        dump_struct_2 = os.path.join(dump_path, "struct_2.csv")
        dump_pairs = os.path.join(dump_path, "pairs.csv")

        struct_1 = pd.read_csv(dump_struct_1, index_col=0)
        struct_2 = pd.read_csv(dump_struct_2, index_col=0)
        pos_pairs_res = pd.read_csv(dump_pairs, index_col=0)

        # Get CA coords
        ca_1 = struct_1[struct_1.name == 'CA']
        ca_2 = struct_2[struct_2.name == 'CA']
        mapping_1 = {resindex: i for i, resindex in enumerate(ca_1.residue.values)}
        mapping_2 = {resindex: i for i, resindex in enumerate(ca_2.residue.values)}

        pos_as_array_1 = np.array([mapping_1[resi] for resi in pos_pairs_res['residue0']])
        pos_as_array_2 = np.array([mapping_2[resi] for resi in pos_pairs_res['residue1']])

        dense = np.zeros((len(ca_1), len(ca_2)))
        dense[pos_as_array_1, pos_as_array_2] = 1
        negs_1, negs_2 = np.where(dense == 0)

        pos_array = np.stack((pos_as_array_1, pos_as_array_2))
        neg_array = np.stack((negs_1, negs_2))
        num_pos = pos_as_array_1.shape[0]
        num_neg = negs_1.shape[0]
        num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)
        pos_array_idx = np.random.choice(pos_array.shape[1], size=num_pos_to_use)
        neg_array_idx = np.random.choice(neg_array.shape[1], size=num_neg_to_use)
        pos_array_sampled = pos_array[:, pos_array_idx]
        neg_array_sampled = neg_array[:, neg_array_idx]

        # Get all coords, extract the right ones and stack them into (n_pairs, 2, 3)
        coords_1 = ca_1[['x', 'y', 'z', ]].values
        coords_2 = ca_2[['x', 'y', 'z', ]].values
        pos_1, pos_2 = coords_1[pos_array_sampled[0]], coords_2[pos_array_sampled[1]]
        pos_stack = np.transpose(np.stack((pos_1, pos_2)), axes=(1, 0, 2))
        neg_1, neg_2 = coords_1[neg_array_sampled[0]], coords_2[neg_array_sampled[1]]
        neg_stack = np.transpose(np.stack((neg_1, neg_2)), axes=(1, 0, 2))
        batch = Data(name1=name1, name2=name2,
                     pos_stack=torch.from_numpy(pos_stack),
                     neg_stack=torch.from_numpy(neg_stack))
        if self.return_surface:
            geom_feats_1 = main.get_diffnetfiles(name=name1,
                                                 df=struct_1,
                                                 dump_surf_dir=self.get_geometry_dir(name1),
                                                 dump_operator_dir=self.get_operator_dir(name1),
                                                 recompute=self.recompute)
            geom_feats_2 = main.get_diffnetfiles(name=name2,
                                                 df=struct_2,
                                                 dump_surf_dir=self.get_geometry_dir(name2),
                                                 dump_operator_dir=self.get_operator_dir(name2),
                                                 recompute=self.recompute)
            if geom_feats_1 is None or geom_feats_2 is None:
                raise ValueError("A geometric feature is buggy")
            batch.geom_feats_1 = geom_feats_1
            batch.geom_feats_2 = geom_feats_2

        if self.return_graph:
            graph_1 = main.get_graph(name=name1, df=struct_1,
                                     dump_graph_dir=self.get_graph_dir(name1),
                                     big=self.big_graphs,
                                     recompute=True)
            graph_2 = main.get_graph(name=name2, df=struct_2,
                                     dump_graph_dir=self.get_graph_dir(name1),
                                     big=self.big_graphs,
                                     recompute=True)
            if graph_1 is None or graph_2 is None:
                raise ValueError("A graph feature is buggy")
            batch.graph_1 = graph_1
            batch.graph_2 = graph_2

        return batch

    def __getitem__(self, index):
        # res = self.get_item(index)
        try:
            res = self.get_item(index)
            return res
        except Exception as e:
            print(f"Error in __getitem__: {e} index : {index}")
            batch = Data()
            return batch


if __name__ == '__main__':
    import time

    t0 = time.perf_counter()
    data_dir = '../data/PIP/DIPS-split/data/test/'
    dataset = NewPIP(data_dir, return_graph=True,
                     geometry_path='../data/processed_data/geometry/',
                     operator_path='../data/processed_data/operator/',
                     graph_path='../data/processed_data/graph',
                     )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, collate_fn=lambda x: x[0])
    for i, res in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # for i, res in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        # print(i)
        if i > 250:
            break
    print(time.perf_counter() - t0)
