import base_nets
import torch
import torch.nn as nn

from atom2d_utils.learning_utils import unwrap_feats, center_normalize


class PSRSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, linear_sizes=(128,), dropout=True,
                 drate=0.3, batch_norm=False, use_xyz=False, use_graph=False):
        super(PSRSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.use_xyz = use_xyz
        # Create the model
        self.use_graph = use_graph
        if not use_graph:
            self.encoder_model = base_nets.layers.DiffusionNet(C_in=in_channels,
                                                               C_out=out_channel,
                                                               C_width=C_width,
                                                               N_block=N_block,
                                                               last_activation=torch.relu)
        else:
            self.encoder_model = base_nets.layers.GraphDiffNet(C_in=in_channels,
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

    def forward(self, x):
        """
        Both inputs should unwrap as (features, confidence, vertices, mass, L, evals, evecs, gradX, gradY, faces)
        pairs_loc are the coordinates of points shape (n_pairs, 2, 3)
        :param x_left:
        :param x_right:
        :return:
        """

        if not self.use_graph:
            geom_feat = x
        else:
            geom_feat, graph = x
        dict_feat = unwrap_feats(geom_feat, device=self.device)

        # We need the vertices to push back the points.
        # We also have to remove them from the dict to feed into base_nets
        verts = dict_feat.pop('vertices')
        if self.use_xyz:
            verts = center_normalize([verts])[0]
            dict_feat["x_in"] = torch.cat([verts, dict_feat["x_in"]], dim=1)

        if not self.use_graph:
            processed = self.encoder_model(**dict_feat)
        else:
            processed = self.encoder_model(graph=graph, vertices=verts, **dict_feat)
        x = torch.max(processed, dim=-2).values
        x = self.top_net(x)
        return x
