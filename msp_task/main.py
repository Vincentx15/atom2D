import os
import sys

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from msp_task import data_loader, models

data_dir = '../data/MSP/train/'
return_graph = True
dataset = data_loader.MSPDataset(data_dir,
                                 geometry_path='../data/MSP/geometry/',
                                 operator_path='../data/MSP/operator/',
                                 graph_path='../data/MSP/graph',
                                 return_graph=return_graph)
model = models.MSPSurfNet(graph_model='bipartite', use_graph=True)
if return_graph:
    data = dataset[0]
    x = (data.geom_feats, data.graph_feats)
    output = model(x, data.coords)
