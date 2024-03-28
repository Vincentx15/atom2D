import os
import sys

import torch
import torch.nn as nn

from base_nets import GraphDiffNetBipartite

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from base_nets.pronet_updated import ProNet


class MasifSiteProNet(torch.nn.Module):
    def __init__(self,
                 level='allatom',
                 num_blocks=4,
                 C_width=128,  # TODO change
                 mid_emb=64,
                 num_radial=6,
                 num_spherical=2,
                 cutoff=10.0,
                 max_num_neighbors=32,
                 int_emb_layers=3,
                 # out_layers=2,
                 num_pos_emb=16,
                 add_seq_emb=False,
                 **kwargs):
        super(MasifSiteProNet, self).__init__()
        hidden_channels = C_width
        self.pronet = ProNet(level=level,
                             num_blocks=num_blocks,
                             hidden_channels=hidden_channels,
                             mid_emb=mid_emb,
                             num_radial=num_radial,
                             num_spherical=num_spherical,
                             cutoff=cutoff,
                             max_num_neighbors=max_num_neighbors,
                             int_emb_layers=int_emb_layers,
                             num_pos_emb=num_pos_emb,
                             add_seq_emb=add_seq_emb)

        self.top_net = nn.Sequential(*[
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_channels, 1)
        ])

    def select_close(self, verts, processed, graph_pos):
        # find nearest neighbors between doing last layers
        with torch.no_grad():
            dists = torch.cdist(verts, graph_pos)
            min_indices = torch.argmin(dists, dim=1)
        selected = processed[min_indices]
        return selected

    def forward(self, batch):
        pronet_graph = batch.pronet_graph
        embed_graph = self.pronet(pronet_graph)
        verts = batch.verts
        processed_feats = embed_graph.split(pronet_graph.batch.bincount().tolist())
        predictions = []
        for vert, feats, graph in zip(verts, processed_feats, pronet_graph.to_data_list()):
            feats_close = self.select_close(vert, feats, graph.coords_ca)
            predictions.append(self.top_net(feats_close))
        return predictions


class MasifSiteNet(torch.nn.Module):
    def __init__(self, in_channels=37, in_channels_surf=25, out_channel=64, C_width=128, N_block=4,
                 with_gradient_features=True, dropout=0.3, batch_norm=False, use_gat=False, use_v2=False,
                 use_skip=False, neigh_th=8, out_features=1, use_distance=False,
                 use_wln=False, add_seq_emb=False, **kwargs):
        super(MasifSiteNet, self).__init__()

        in_channels = in_channels + 1280 if add_seq_emb else in_channels
        self.in_channels = in_channels + 1280 if add_seq_emb else in_channels

        self.in_channels_surf = in_channels_surf
        self.out_channel = out_channel

        # Create the model
        self.encoder_model = GraphDiffNetBipartite(C_in_graph=in_channels,
                                                   C_in_surf=in_channels_surf,
                                                   C_out=out_channel,
                                                   C_width=C_width,
                                                   N_block=N_block,
                                                   with_gradient_features=with_gradient_features,
                                                   last_activation=torch.relu,
                                                   use_bn=batch_norm,
                                                   output_graph=False,
                                                   use_gat=use_gat,
                                                   use_v2=use_v2,
                                                   use_skip=use_skip,
                                                   use_wln=use_wln,
                                                   neigh_th=neigh_th,
                                                   dropout=dropout,
                                                   use_distance=use_distance)
        # Top FCs
        in_features = out_channel
        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features // 2, out_features=out_features)
        ])

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, batch):
        surface = batch.surface
        graph = batch.graph
        processed = self.encoder_model(graph=graph, surface=surface)
        return [self.top_net(proc) for proc in processed]
