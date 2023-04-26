import diff_net
import torch
import torch.nn as nn

from atom2d_utils.learning_utils import unwrap_feats
from data_processing.point_cloud_utils import torch_rbf

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


def create_pyg_graph_object(coords, features, sigma=4):
    num_nodes = len(coords)

    # Calculate pairwise distances using torch.cdist
    with torch.no_grad():
        pairwise_distances = torch.cdist(coords, coords)
        rbf_weights = torch.exp(-pairwise_distances / sigma)

        # Create edge index using torch.triu_indices and remove self-loops
        row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
        edge_index = torch.stack([row, col], dim=0)

        # Create bidirectional edges by concatenating (row, col) and (col, row)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Extract edge weights from pairwise_distances using the created edge_index
        edge_weight = rbf_weights[row, col]
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    return Data(x=features, edge_index=edge_index, edge_weight=edge_weight)


# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channel, drate=None):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channel)
        self.drate = drate

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        if self.drate is not None:
            x = F.dropout(x, p=self.drate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def get_mlp(in_features, hidden_sizes, batch_norm=True, drate=None):
    layers = []
    for units in hidden_sizes:
        layers.extend([
            nn.Linear(in_features, units),
            nn.ReLU()
        ])
        if batch_norm:
            layers.append(nn.BatchNorm1d(units))
        if drate is not None:
            layers.append(nn.Dropout(drate))
        in_features = units

    # Final FC layer
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


class MSPSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, hidden_sizes=(128,), drate=0.3, batch_norm=False):
        super(MSPSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        # Create the model
        self.diff_net_model = diff_net.layers.DiffusionNet(C_in=in_channels,
                                                           C_out=out_channel,
                                                           C_width=10,
                                                           last_activation=torch.relu)
        self.gcn = GCN(num_features=2 * (out_channel + 1), hidden_channels=out_channel, out_channel=out_channel,
                       drate=drate)
        self.top_mlp = get_mlp(in_features=out_channel,
                               hidden_sizes=hidden_sizes,
                               drate=drate,
                               batch_norm=batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, coords):
        """
        Here inputs are [left_orig, right_orig, left_mut, right_mut]

        :param x:
        :param coords: orig_coords, mut_coords
        :return:
        """

        all_dict_feat = [unwrap_feats(geom_feat, device=self.device) for geom_feat in x]
        vertices = [dict_feat.pop('vertices') for dict_feat in all_dict_feat]
        # We need the vertices to push back the points.
        # We also have to remove them from the dict to feed into diff_net
        processed = [self.diff_net_model(**dict_feat) for dict_feat in all_dict_feat]

        # Now project each part onto its points
        projected_left_orig = torch_rbf(points_1=vertices[0], feats_1=processed[0], points_2=coords[0], concat=True)
        projected_right_orig = torch_rbf(points_1=vertices[1], feats_1=processed[1], points_2=coords[0], concat=True)
        projected_left_mut = torch_rbf(points_1=vertices[2], feats_1=processed[2], points_2=coords[1], concat=True)
        projected_right_mut = torch_rbf(points_1=vertices[3], feats_1=processed[3], points_2=coords[1], concat=True)
        projected_orig = torch.cat((projected_left_orig, projected_right_orig), dim=-1)
        projected_mut = torch.cat((projected_left_mut, projected_right_mut), dim=-1)

        # Example coordinates and features
        orig_graph = create_pyg_graph_object(coords[0], projected_orig)
        mut_graph = create_pyg_graph_object(coords[1], projected_mut)

        orig_nodes = self.gcn(orig_graph)
        mut_nodes = self.gcn(mut_graph)
        x = torch.cat((orig_nodes, mut_nodes), dim=-2)  # TODO
        x = torch.mean(x, dim=-2)  # meanpool
        x = self.top_mlp(x)
        x = torch.sigmoid(x)
        return x
