from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from base_nets import DiffusionNet, GraphDiffNet, GraphDiffNetSequential, GraphDiffNetAttention, GraphDiffNetBipartite, AtomNetGraph

from data_processing import point_cloud_utils
from atom2d_utils.learning_utils import unwrap_feats, center_normalize


class PIPNet(torch.nn.Module):
    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, dropout=0.3, batch_norm=False, sigma=2.5,
                 use_xyz=False, use_graph=False, use_graph_only=False, clip_output=False, graph_model='parallel',
                 **kwargs):
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
            self.encoder_model = AtomNetGraph(C_in=in_channels,
                                              C_out=out_channel,
                                              C_width=C_width)
            self.fc1 = nn.Linear(C_width * 4, C_width * 4)
            self.fc2 = nn.Linear(C_width * 4, 1)
        elif use_graph:
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
        else:
            self.encoder_model = DiffusionNet(C_in=in_channels,
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

    @property
    def device(self):
        return next(self.parameters()).device

    def project_processed_graph(self, locs_left, locs_right, processed_left, processed_right, graph_left, graph_right):
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
        return x

    def project_processed_surface(self, locs_left, locs_right, processed_left, processed_right, verts_left,
                                  verts_right):
        # Push this signal onto the CA locations
        feats_left = point_cloud_utils.torch_rbf(points_1=verts_left, feats_1=processed_left,
                                                 points_2=locs_left, concat=True, sigma=self.sigma)
        feats_right = point_cloud_utils.torch_rbf(points_1=verts_right, feats_1=processed_right,
                                                  points_2=locs_right, concat=True, sigma=self.sigma)

        # Once equiped with the features and confidence scores at each point, feed that into the networks
        # no need for sigmoid since we use BCEWithLogitsLoss
        x = torch.cat([feats_left, feats_right], dim=1)
        x = self.top_net(x)
        return x

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
            data = batch[0]
            pairs_loc = torch.cat((data.pos_stack, data.neg_stack), dim=-3)
            locs_left, locs_right = pairs_loc[..., 0, :].float(), pairs_loc[..., 1, :].float()

            if not self.use_graph_only:
                dict_feat_left = unwrap_feats(data.geom_feats_1, device=self.device)
                dict_feat_right = unwrap_feats(data.geom_feats_2, device=self.device)
                verts_left = dict_feat_left.pop('vertices')
                verts_right = dict_feat_right.pop('vertices')
                if self.use_xyz:
                    verts_leftt, locs_leftt = center_normalize([verts_left], [locs_left])
                    verts_rightt, locs_rightt = center_normalize([verts_right], [locs_right])
                    verts_leftt, locs_left = verts_leftt[0], locs_leftt[0]
                    verts_rightt, locs_right = verts_rightt[0], locs_rightt[0]
                    rot_mat = torch.from_numpy(R.random().as_matrix()).float().to(self.device)  # random rotation
                    verts_rightt = verts_rightt @ rot_mat  # random rotation
                    locs_right = locs_right @ rot_mat  # random rotation
                    x_in1, x_in2 = dict_feat_left["x_in"], dict_feat_right["x_in"]
                    dict_feat_left["x_in"] = torch.cat([verts_leftt, x_in1], dim=1)
                    dict_feat_right["x_in"] = torch.cat([verts_rightt, x_in2], dim=1)
            if self.use_graph:
                graph_left = data.graph_1
                graph_right = data.graph_2

            if not self.use_graph:
                processed_left = self.encoder_model(**dict_feat_left)
                processed_right = self.encoder_model(**dict_feat_right)
            elif self.use_graph_only:
                processed_left = self.encoder_model(graph=graph_left)
                processed_right = self.encoder_model(graph=graph_right)
            else:
                processed_left = self.encoder_model(graph=graph_left, vertices=verts_left, **dict_feat_left)
                processed_right = self.encoder_model(graph=graph_right, vertices=verts_right, **dict_feat_right)

            # Once processed, project back onto the query points
            if self.use_graph_only:
                x = self.project_processed_graph(locs_left, locs_right,
                                                 processed_left, processed_right,
                                                 graph_left, graph_right)
            else:
                x = self.project_processed_surface(locs_left, locs_right,
                                                   processed_left, processed_right,
                                                   verts_left, verts_right)
            result = x.view(-1)
            if self.clip_output:
                # clip the values such that after applying sigmoid we get 0.01 and 0.99
                result = torch.clamp(result, min=-4.6, max=4.6)
        else:
            all_graphs_left = list()
            all_graphs_right = list()
            all_locs_left = list()
            all_locs_right = list()

            for data in batch:
                pairs_loc = torch.cat((data.pos_stack, data.neg_stack), dim=-3)
                locs_left, locs_right = pairs_loc[..., 0, :].float(), pairs_loc[..., 1, :].float()
                all_locs_left.append(locs_left)
                all_locs_right.append(locs_right)
                # TODO investigate : the following line are needed to fill processed later on...
                data.graph_1.processed = torch.zeros(data.graph_1.num_nodes, 1)
                data.graph_2.processed = torch.zeros(data.graph_2.num_nodes, 1)
                all_graphs_left.append(data.graph_1)
                all_graphs_right.append(data.graph_2)

            # Batch graphs left for efficient encoding.
            graphs_left = Batch.from_data_list(all_graphs_left)
            graphs_right = Batch.from_data_list(all_graphs_right)
            # TODO investigate : the following line does not work as it is not split when to_list is called
            # maybe only the keys present at batch time are tracked... This works anyway
            graphs_left.processed = self.encoder_model(graph=graphs_left)
            graphs_right.processed = self.encoder_model(graph=graphs_right)

            # Now split back everything and project
            all_graphs_left = graphs_left.to_data_list()
            all_graphs_right = graphs_right.to_data_list()
            all_preds = []
            for locs_left, locs_right, graph_left, graph_right in zip(all_locs_left, all_locs_right,
                                                                      all_graphs_left, all_graphs_right):
                processed_left = graph_left.processed
                processed_right = graph_right.processed
                # Once processed, project back onto the query points
                x = self.project_processed_graph(locs_left, locs_right,
                                                 processed_left, processed_right,
                                                 graph_left, graph_right)
                all_preds.append(x)
            result = torch.cat(all_preds)
            result = result.view(-1)
            if self.clip_output:
                # clip the values such that after applying sigmoid we get 0.01 and 0.99
                result = torch.clamp(result, min=-4.6, max=4.6)

        return result
