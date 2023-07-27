import base_nets
import torch
import torch.nn as nn
import torch.nn.functional as F

from atom2d_utils.learning_utils import unwrap_feats, center_normalize


class PSRSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, linear_sizes=(128,), dropout=True,
                 drate=0.3, batch_norm=False, use_xyz=False, use_graph=False, use_graph_only=False,
                 graph_model='parallel', **kwargs):
        super(PSRSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.use_xyz = use_xyz
        # Create the model
        self.use_graph = use_graph or use_graph_only
        self.use_graph_only = use_graph_only
        if use_graph_only:
            self.encoder_model = base_nets.layers.AtomNetGraph(C_in=in_channels,
                                                               C_out=out_channel,
                                                               C_width=C_width,
                                                               last_factor=4)
            self.fc1 = nn.Linear(C_width * 4, C_width * 2)
            self.fc2 = nn.Linear(C_width * 2, 1)
        elif not use_graph:
            self.encoder_model = base_nets.layers.DiffusionNet(C_in=in_channels,
                                                               C_out=out_channel,
                                                               C_width=C_width,
                                                               N_block=N_block,
                                                               last_activation=torch.relu)
        else:
            if graph_model == 'parallel':
                self.encoder_model = base_nets.layers.GraphDiffNet(C_in=in_channels,
                                                                   C_out=out_channel,
                                                                   C_width=C_width,
                                                                   N_block=N_block,
                                                                   last_activation=torch.relu)
            elif graph_model == 'sequential':
                self.encoder_model = base_nets.layers.GraphDiffNetSequential(C_in=in_channels,
                                                                             C_out=out_channel,
                                                                             C_width=C_width,
                                                                             N_block=N_block,
                                                                             last_activation=torch.relu)
            elif graph_model == 'attention':
                self.encoder_model = base_nets.layers.GraphDiffNetAttention(C_in=in_channels,
                                                                            C_out=out_channel,
                                                                            C_width=C_width,
                                                                            N_block=N_block,
                                                                            last_activation=torch.relu)
            elif graph_model == 'bipartite':
                self.encoder_model = base_nets.layers.GraphDiffNetAttention(C_in=in_channels,
                                                                            C_out=out_channel,
                                                                            C_width=C_width,
                                                                            N_block=N_block,
                                                                            last_activation=torch.relu)

        # This corresponds to each averaged embedding and confidence scores for each pair of CA
        layers = []
        # Top FCs
        in_features = out_channel
        for units in linear_sizes:
            layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU()
            ])
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            if dropout:
                layers.append(nn.Dropout(drate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))
        self.top_net = nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch):
        """
        Both inputs should unwrap as (features, confidence, vertices, mass, L, evals, evecs, gradX, gradY, faces)
        pairs_loc are the coordinates of points shape (n_pairs, 2, 3)
        :param x_left:
        :param x_right:
        :return:
        """

        assert len(batch) == 1 or self.use_graph_only

        if not self.use_graph_only:
            dict_feat = unwrap_feats(batch[0].geom_feats, device=self.device)
            verts = dict_feat.pop('vertices')
            if self.use_xyz:
                # We need the vertices to push back the points.
                # We also have to remove them from the dict to feed into base_nets
                verts = center_normalize([verts])[0]
                dict_feat["x_in"] = torch.cat([verts, dict_feat["x_in"]], dim=1)
        if self.use_graph:
            all_graphs = [data.graph_feat for data in batch]
            from torch_geometric.data import Batch
            graph = Batch.from_data_list(all_graphs)

        if not self.use_graph:
            processed = self.encoder_model(**dict_feat)
        elif self.use_graph_only:
            processed = self.encoder_model(graph=graph)
        else:
            processed = self.encoder_model(graph=graph, vertices=verts, **dict_feat)

        if self.use_graph_only:
            graph.x = processed
            graph_embs = []
            for individual_graph in graph.to_data_list():
                x = torch.max(individual_graph.x, dim=-2).values
                x = F.relu(x)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, p=0.25, training=self.training)
                x = self.fc2(x).view(-1)
                graph_embs.append(x)
            return torch.cat(graph_embs)
        else:
            x = torch.max(processed, dim=-2).values
            x = self.top_net(x)
        return x
