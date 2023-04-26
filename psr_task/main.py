import os
import sys

import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from psr_task import data_loader, models

data_dir = '../data/PSR/train/'
dataset = data_loader.PSRDataset(data_dir)
model = models.PSRSurfNet()
name, geom_feats, scores = dataset[0]
model(geom_feats)
