import torch
import torch.nn as nn

from base_nets import DiffusionNetBatch, GraphDiffNetParallel, GraphDiffNetSequential, GraphDiffNetAttention, \
    GraphDiffNetBipartite, AtomNetGraph

from data_processing import point_cloud_utils


class PIPNet(torch.nn.Module):
    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, dropout=0.3, batch_norm=False, sigma=2.5,
                 use_graph=False, use_graph_only=False, clip_output=False, graph_model='parallel', output_graph=False,
                 use_gat=False, use_v2=False, use_skip=False, neigh_th=8, flash=True, use_distance=False, use_mp=False,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.sigma = sigma
        self.use_gat = use_gat
        # Create the model
        self.use_graph = use_graph or use_graph_only
        self.use_graph_only = use_graph_only
        self.output_graph = output_graph
        self.clip_output = clip_output
        if use_graph_only:
            self.encoder_model = AtomNetGraph(C_in=in_channels,
                                              C_out=out_channel,
                                              C_width=C_width,
                                              use_distance=use_distance)
        elif not use_graph:
            self.encoder_model = DiffusionNetBatch(C_in=5,
                                                   C_out=out_channel,
                                                   C_width=C_width,
                                                   N_block=N_block,
                                                   last_activation=torch.relu,
                                                   use_bn=batch_norm)
        else:
            if graph_model == 'parallel':
                self.encoder_model = GraphDiffNetParallel(C_in=in_channels,
                                                          C_out=out_channel,
                                                          C_width=C_width,
                                                          N_block=N_block,
                                                          last_activation=torch.relu,
                                                          use_bn=batch_norm,
                                                          use_mp=False,
                                                          output_graph=output_graph,
                                                          use_distance=use_distance)
            elif graph_model == 'sequential':
                self.encoder_model = GraphDiffNetSequential(C_in=in_channels,
                                                            C_out=out_channel,
                                                            C_width=C_width,
                                                            N_block=N_block,
                                                            last_activation=torch.relu,
                                                            use_bn=batch_norm,
                                                            use_mp=use_mp,
                                                            use_skip=use_skip,
                                                            use_gat=use_gat,
                                                            output_graph=output_graph,
                                                            use_distance=use_distance)
            elif graph_model == 'attention':
                self.encoder_model = GraphDiffNetAttention(C_in=in_channels,
                                                           C_out=out_channel,
                                                           C_width=C_width,
                                                           N_block=N_block,
                                                           last_activation=torch.relu,
                                                           use_bn=batch_norm,
                                                           output_graph=output_graph,
                                                           flash=flash,
                                                           use_distance=use_distance)
            elif graph_model == 'bipartite':
                self.encoder_model = GraphDiffNetBipartite(C_in_graph=in_channels,
                                                           C_out=out_channel,
                                                           C_width=C_width,
                                                           N_block=N_block,
                                                           last_activation=torch.relu,
                                                           use_bn=batch_norm,
                                                           output_graph=output_graph,
                                                           use_gat=use_gat,
                                                           use_v2=use_v2,
                                                           use_skip=use_skip,
                                                           neigh_th=neigh_th,
                                                           use_distance=use_distance)

        if self.use_graph_only:
            in_features = C_width * 4
        elif output_graph:
            in_features = 2 * out_channel
        else:
            in_features = 2 * (out_channel + 1)

        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features, 1)
        ])

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
        x = self.top_net(x)
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

    def forward(self, batch, return_embs=False):
        """
        Both inputs should unwrap as (features, confidence, vertices, mass, L, evals, evecs, gradX, gradY, faces)
        pairs_loc are the coordinates of points shape (n_pairs, 2, 3)
        :param x_left:
        :param x_right:
        :return:
        """
        graph_1, graph_2, surface_1, surface_2 = None, None, None, None
        locs_left, locs_right = batch.locs_left, batch.locs_right

        if not self.use_graph_only:
            surface_1, surface_2 = batch.surface_1, batch.surface_2
            verts_left, verts_right = surface_1.vertices, surface_2.vertices
        if self.use_graph:
            graph_1, graph_2 = batch.graph_1, batch.graph_2

        # forward pass
        processed_left = self.encoder_model(graph=graph_1, surface=surface_1)
        processed_right = self.encoder_model(graph=graph_2, surface=surface_2)
        if self.use_graph_only:
            processed_left = processed_left.split(graph_1.batch.bincount().tolist())
            processed_right = processed_right.split(graph_2.batch.bincount().tolist())

        xs = []
        if self.use_graph_only or (self.use_graph and self.output_graph):
            for loc_left, loc_right, proc_left, proc_right, g_left, g_right in zip(locs_left, locs_right,
                                                                                   processed_left, processed_right,
                                                                                   graph_1.to_data_list(),
                                                                                   graph_2.to_data_list()):
                if return_embs:
                    return proc_left, proc_right, g_left, g_right,
                xs.append(self.project_processed_graph(loc_left, loc_right, proc_left, proc_right, g_left, g_right))
        else:
            for loc_left, loc_right, proc_left, proc_right, g_left, g_right in zip(locs_left, locs_right,
                                                                                   processed_left, processed_right,
                                                                                   verts_left, verts_right):
                xs.append(self.project_processed_surface(loc_left, loc_right, proc_left, proc_right, g_left, g_right))

        result = torch.cat(xs)
        result = result.view(-1)

        # clip the values such that after applying sigmoid we get 0.01 and 0.99
        if self.clip_output:
            result = torch.clamp(result, min=-4.6, max=4.6)

        return result
