import diffusion_net
import torch
import torch.nn as nn

from atom2d_utils.learning_utils import unwrap_feats


class PSRSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, linear_sizes=(128,), dropout=True, drate=0.3, batch_norm=False):
        super(PSRSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        # Create the model
        self.diff_net_model = diffusion_net.layers.DiffusionNet(C_in=in_channels,
                                                                C_out=out_channel,
                                                                C_width=10,
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

        dict_feat_left = unwrap_feats(x, device=self.device)

        # We need the vertices to push back the points.
        # We also have to remove them from the dict to feed into diff_net
        _ = dict_feat_left.pop('vertices')

        processed = self.diff_net_model(**dict_feat_left)
        x = torch.max(processed, dim=-2).values
        x = self.top_net(x)
        return x
