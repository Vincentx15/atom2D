import torch
import torch.nn as nn

from base_nets import DiffusionNetBatch, GraphDiffNet, GraphDiffNetSequential, GraphDiffNetAttention, GraphDiffNetBipartite, AtomNetGraph


class PSRSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, linear_sizes=(128,), dropout=True,
                 drate=0.3, batch_norm=False, use_graph=False, use_graph_only=False,
                 graph_model='parallel', **kwargs):
        super(PSRSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        # Create the model
        self.use_graph = use_graph or use_graph_only
        self.use_graph_only = use_graph_only
        if use_graph_only:
            self.encoder_model = AtomNetGraph(C_in=in_channels,
                                              C_out=out_channel,
                                              C_width=C_width,
                                              last_factor=4)
            self.top_net_graph = nn.Sequential(*[
                nn.ReLU(),
                nn.Linear(C_width * 4, C_width * 2),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(C_width * 2, 1)
            ])
        elif not use_graph:
            self.encoder_model = DiffusionNetBatch(C_in=in_channels,
                                                   C_out=out_channel,
                                                   C_width=C_width,
                                                   N_block=N_block,
                                                   last_activation=torch.relu)
        else:
            if graph_model == 'parallel':
                self.encoder_model = GraphDiffNet(C_in=in_channels,
                                                  C_out=out_channel,
                                                  C_width=C_width,
                                                  N_block=N_block,
                                                  last_activation=torch.relu)
            elif graph_model == 'sequential':
                self.encoder_model = GraphDiffNetSequential(C_in=in_channels,
                                                            C_out=out_channel,
                                                            C_width=C_width,
                                                            N_block=N_block,
                                                            last_activation=torch.relu)
            elif graph_model == 'attention':
                self.encoder_model = GraphDiffNetAttention(C_in=in_channels,
                                                           C_out=out_channel,
                                                           C_width=C_width,
                                                           N_block=N_block,
                                                           last_activation=torch.relu)
            elif graph_model == 'bipartite':
                self.encoder_model = GraphDiffNetBipartite(C_in=in_channels,
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

        graph, surface = None, None

        if not self.use_graph_only:
            surface = batch.surface
        if self.use_graph:
            graph = batch.graph

        if not self.use_graph:  # todo add surface to other model
            processed = self.encoder_model(graph=graph, surface=surface)
        elif self.use_graph_only:
            processed = self.encoder_model(graph=graph, surface=surface)
        else:
            processed = self.encoder_model(graph=graph, surface=surface)  # , vertices=verts, **dict_feat)

        if self.use_graph_only:
            graph.x = processed
            graph_embs = []
            for individual_graph in graph.to_data_list():
                x = torch.max(individual_graph.x, dim=-2).values
                x = self.top_net_graph(x).view(-1)
                graph_embs.append(x)
            return torch.cat(graph_embs)
        else:
            x = torch.stack([torch.max(x, dim=-2).values for x in processed])
            x = self.top_net(x)
        return x
