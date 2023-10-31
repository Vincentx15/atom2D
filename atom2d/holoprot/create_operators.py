import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.get_operators import surf_to_operators


class HoloProtPreprocess(torch.utils.data.Dataset):
    def __init__(self, surface_path, operator_path):
        super().__init__()
        self.surface_path = surface_path
        self.operator_path = operator_path
        os.makedirs(operator_path, exist_ok=True)
        self.surfaces = os.listdir(self.surface_path)
        self.failed = 0

    def __len__(self):
        return len(self.surfaces)

    def __getitem__(self, idx):
        """
        Return a list of subunit for this item.
        :param index:
        :return:
        """
        surface = self.surfaces[idx]
        surf_path = os.path.join(self.surface_path, surface)
        operator_path = os.path.join(self.operator_path, surface.replace('.pth', '_.npz'))
        try:
            surface_data = torch.load(surf_path)['prot']
            operators = surf_to_operators(npz_path=operator_path,
                                          vertices=surface_data.vertices,
                                          faces=surface_data.faces)
        except Exception as e:
            self.failed += 1
            print(surface, self.failed, e)
        return 0


if __name__ == '__main__':
    pass
    proc_dataset = HoloProtPreprocess(surface_path="../../data/holoprot/datasets/processed/enzyme/surface",
                                      operator_path="../../data/holoprot/datasets/processed/enzyme/operators")
    proc_loader = torch.utils.data.DataLoader(proc_dataset, num_workers=6, collate_fn=lambda x: x)
    for i, x in enumerate(proc_loader):
        if not i % 200:
            print("Done", i)
