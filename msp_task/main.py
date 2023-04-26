import os
import sys

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from msp_task import data_loader, models

data_dir = '../data/MSP/train/'
dataset = data_loader.MSPDataset(data_dir)
model = models.MSPSurfNet()
name, geom_feats, coords, label = dataset[0]
model(geom_feats, coords)
