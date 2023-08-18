import os
import sys

from atom3d.datasets import LMDBDataset
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from atom2d_utils import naming_utils


class Atom3DDataset(torch.utils.data.Dataset):
    """
    Generic class for thing that contain a LMDB dataset and interact with precomputed data
    - Processor dataset
    - Learning dataset
    """

    def __init__(self, lmdb_path, geometry_path, operator_path, graph_path=None):
        self._lmdb_dataset = LMDBDataset(lmdb_path)
        self.lmdb_path = lmdb_path
        self.geometry_path = geometry_path
        self.operator_path = operator_path
        self.graph_path = graph_path
        self.failed_set = set()

    def __len__(self) -> int:
        return len(self._lmdb_dataset)

    def get_geometry_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.geometry_path)

    def get_operator_dir(self, name):
        return naming_utils.name_to_dir(name, dir_path=self.operator_path)

    def get_graph_dir(self, name):
        if self.graph_path is None:
            raise ValueError("Asking for graphs while graph_path is not set")
        return naming_utils.name_to_dir(name, dir_path=self.graph_path)
