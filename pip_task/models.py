import base_nets
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_processing import point_cloud_utils
from atom2d_utils.learning_utils import unwrap_feats, center_normalize


class PIPNet(torch.nn.Module):
    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, dropout=0.3, batch_norm=False, sigma=2.5,
                 use_xyz=False, use_graph=False, use_graph_only=False, clip_output=False, graph_model='parallel', **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.sigma = sigma
        self.use_xyz = use_xyz
        # Create the model
        self.use_graph = use_graph or use_graph_only
        self.use_graph_only = use_graph_only
        self.clip_output = clip_output
        if use_graph_only:
            self.encoder_model = base_nets.layers.AtomNetGraph(C_in=in_channels,
                                                               C_out=out_channel,
                                                               C_width=C_width)
            self.fc1 = nn.Linear(C_width * 4, C_width * 4)
            self.fc2 = nn.Linear(C_width * 4, 1)
        elif use_graph:
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
        else:
            self.encoder_model = base_nets.layers.DiffusionNet(C_in=in_channels,
                                                               C_out=out_channel,
                                                               C_width=C_width,
                                                               N_block=N_block,
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
                layers.append(nn.Dropout(dropout))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))
        self.top_net = nn.Sequential(*layers)

    def forward(self, x_left, x_right, pairs_loc):
        """
        Both inputs should unwrap as (features, confidence, vertices, mass, L, evals, evecs, gradX, gradY, faces)
        pairs_loc are the coordinates of points shape (n_pairs, 2, 3)
        :param x_left:
        :param x_right:
        :return:
        """
        device = pairs_loc.device
        locs_left, locs_right = pairs_loc[..., 0, :].float(), pairs_loc[..., 1, :].float()

        if self.use_graph:
            x_left, graph_left = x_left
            x_right, graph_right = x_right
        dict_feat_left = unwrap_feats(x_left, device=device)
        dict_feat_right = unwrap_feats(x_right, device=device)
        verts_left = dict_feat_left.pop('vertices')
        verts_right = dict_feat_right.pop('vertices')

        if self.use_xyz:
            verts_leftt, locs_leftt = center_normalize([verts_left], [locs_left])
            verts_rightt, locs_rightt = center_normalize([verts_right], [locs_right])
            verts_leftt, locs_left = verts_leftt[0], locs_leftt[0]
            verts_rightt, locs_right = verts_rightt[0], locs_rightt[0]
            rot_mat = torch.from_numpy(R.random().as_matrix()).float().to(device)  # random rotation
            verts_rightt = verts_rightt @ rot_mat  # random rotation
            locs_right = locs_right @ rot_mat  # random rotation
            x_in1, x_in2 = dict_feat_left["x_in"], dict_feat_right["x_in"]
            dict_feat_left["x_in"] = torch.cat([verts_leftt, x_in1], dim=1)
            dict_feat_right["x_in"] = torch.cat([verts_rightt, x_in2], dim=1)

        if self.use_graph:
            processed_left = self.encoder_model(graph=graph_left, vertices=verts_left, **dict_feat_left)
            processed_right = self.encoder_model(graph=graph_right, vertices=verts_right, **dict_feat_right)
        else:
            processed_left = self.encoder_model(**dict_feat_left)
            processed_right = self.encoder_model(**dict_feat_right)

        if self.use_graph_only:
            # find nearest neighbors between doing last layers
            dists = torch.cdist(locs_left, graph_left.pos)
            min_indices = torch.argmin(dists, dim=1)
            processed_left = processed_left[min_indices]
            dists = torch.cdist(locs_right, graph_right.pos)
            min_indices = torch.argmin(dists, dim=1)
            processed_right = processed_right[min_indices]

            x = torch.cat((processed_left, processed_right), dim=1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.25, training=self.training)
            x = self.fc2(x)
            # no need for sigmoid since we use BCEWithLogitsLoss
        else:
            # TODO remove double from loading... probably also in the dumping

            # Push this signal onto the CA locations
            feats_left = point_cloud_utils.torch_rbf(points_1=verts_left, feats_1=processed_left,
                                                     points_2=locs_left, concat=True, sigma=self.sigma)
            feats_right = point_cloud_utils.torch_rbf(points_1=verts_right, feats_1=processed_right,
                                                      points_2=locs_right, concat=True, sigma=self.sigma)

            # Once equiped with the features and confidence scores at each point, feed that into the networks
            x = torch.cat([feats_left, feats_right], dim=1)
            x = self.top_net(x)
            # no need for sigmoid since we use BCEWithLogitsLoss
            # result = torch.sigmoid(x).view(-1)

        result = x.view(-1)

        if self.clip_output:
            # clip the values such that after applying sigmoid we get 0.01 and 0.99
            result = torch.clamp(result, min=-4.6, max=4.6)
        return result
