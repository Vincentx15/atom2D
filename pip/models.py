import diffusion_net
import torch
import torch.nn as nn

from data_processing import point_cloud_utils


class SurfNet(torch.nn.Module):
    def __init__(self, in_channels=5, out_channel=64, dropout=True, drate=0.3, batch_norm=False):
        super(SurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        # Create the model
        self.diff_net_model = diffusion_net.layers.DiffusionNet(C_in=in_channels,
                                                                C_out=out_channel,
                                                                C_width=10,
                                                                last_activation=torch.relu)
        # This corresponds to each averaged embedding and confidence scores for each pair of CA
        in_features = 2 * (out_channel + 1)
        layers = []
        # Top FCs
        for units in [128] * 2:
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

    def forward(self, x_left, x_right, pairs_loc):
        """
        Both inputs should unwrap as (features, confidence, mass, L, evals, evecs, gradX, gradY, faces)
        pairs_loc are the coordinates of points shape (n_pairs, 2, 3)
        :param x_left:
        :param x_right:
        :return:
        """
        device = pairs_loc.device

        def unwrap_feats(geom_feat):
            features, confidence, vertices, mass, L, evals, evecs, gradX, gradY, faces = geom_feat
            features = torch.cat((features, confidence[..., None]), dim=-1)
            gradX, gradY = gradX.to_sparse(), gradY.to_sparse()
            dict_return = {'x_in': features,
                           'mass': mass,
                           'L': L,
                           'evals': evals,
                           'evecs': evecs,
                           'gradX': gradX,
                           'gradY': gradY,
                           'faces': faces}
            dict_return_32 = {k: v.float().to(device) for k, v in dict_return.items()}
            return dict_return_32

        processed_left = self.diff_net_model(**unwrap_feats(x_left))
        processed_right = self.diff_net_model(**unwrap_feats(x_right))

        # TODO remove double from loading... probably also in the dumping
        # Push this signal onto the CA locations
        locs_left, locs_right = pairs_loc[..., 0, :].float(), pairs_loc[..., 1, :].float()
        verts_left = x_left[2].float().to(device)
        verts_right = x_right[2].float().to(device)
        feats_left, confidence_left = point_cloud_utils.torch_rbf(points_1=verts_left, feats_1=processed_left,
                                                                  points_2=locs_left)
        feats_right, confidence_right = point_cloud_utils.torch_rbf(points_1=verts_right, feats_1=processed_right,
                                                                    points_2=locs_right)

        # Once equiped with the features and confidence scores at each point, feed that into the networks
        x = torch.cat([feats_left, confidence_left[..., None], feats_right, confidence_right[..., None]], dim=1)
        x = self.top_net(x)
        result = torch.sigmoid(x).view(-1)
        return result
