import torch
import torch.nn as nn

from base_nets import GraphDiffNetBipartite


class MasifSiteNet(torch.nn.Module):
    def __init__(self, in_channels=37, in_channels_surf=25, out_channel=64, C_width=128, N_block=4,
                 with_gradient_features=True, dropout=0.3, batch_norm=False, use_gat=False, use_v2=False,
                 use_skip=False, neigh_th=8, out_features=1, use_distance=False,
                 use_wln=False, **kwargs):
        super(MasifSiteNet, self).__init__()

        self.in_channels = in_channels
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
