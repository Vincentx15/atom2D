import torch
import torch.nn as nn
from torch_geometric.data import Batch

from atom2d_utils.learning_utils import unwrap_feats, center_normalize
from base_nets import DiffusionNet, GraphDiffNet, GraphDiffNetSequential, GraphDiffNetAttention, GraphDiffNetBipartite, AtomNetGraph, GCN
from base_nets.diffusion_net.layers import get_mlp
from base_nets.utils import create_pyg_graph_object
from data_processing.point_cloud_utils import torch_rbf


class MSPSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, hidden_sizes=(128,), drate=0.3,
                 batch_norm=False, use_max=True, use_mean=False, use_xyz=False, use_graph=False, use_graph_only=False,
                 graph_model='parallel', **kwargs):
        super(MSPSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.use_max = use_max
        self.use_mean = use_mean
        self.use_xyz = use_xyz

        # Create the model
        self.use_graph = use_graph or use_graph_only
        self.use_graph_only = use_graph_only
        if use_graph_only:
            self.encoder_model = AtomNetGraph(C_in=in_channels,
                                              C_out=out_channel,
                                              C_width=C_width)
            self.fc1 = nn.Linear(C_width * 4, C_width * 4)
            self.fc2 = nn.Linear(C_width * 4, 1)
        elif not use_graph:
            self.encoder_model = DiffusionNet(C_in=in_channels,
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
        infeature_gcn = 2 * (out_channel + 1) if not self.use_graph_only else C_width * 2
        self.gcn = GCN(num_features=infeature_gcn, hidden_channels=out_channel, out_channel=out_channel,
                       drate=drate)
        self.top_mlp = get_mlp(in_features=2 * out_channel,
                               hidden_sizes=hidden_sizes,
                               drate=drate,
                               batch_norm=batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def project_processed_surface(self, vertices, processed, coords):
        """
        :param all_graphs: list of 4 xyz orig_left, orig_right, mut_left, mut_right
        :param processed: same list with their node encodings
        :param coords: list of 2 coords : orig and mut
        :return:
        """
        projected_left_orig = torch_rbf(points_1=vertices[0], feats_1=processed[0], points_2=coords[0], concat=True)
        projected_right_orig = torch_rbf(points_1=vertices[1], feats_1=processed[1], points_2=coords[0],
                                         concat=True)
        projected_left_mut = torch_rbf(points_1=vertices[2], feats_1=processed[2], points_2=coords[1], concat=True)
        projected_right_mut = torch_rbf(points_1=vertices[3], feats_1=processed[3], points_2=coords[1], concat=True)
        projected_orig = torch.cat((projected_left_orig, projected_right_orig), dim=-1)
        projected_mut = torch.cat((projected_left_mut, projected_right_mut), dim=-1)
        return coords[0], coords[1], projected_orig, projected_mut

    def project_processed_graph(self, all_graphs, processed, coords):
        """

        :param all_graphs: list of 4 graphs  orig_left, orig_right, mut_left, mut_right
        :param processed: same list with their node encodings
        :param coords: list of 2 coords : orig and mut
        :return:
        """
        coords_lo, feat_lo = find_nn_feat(coords[0], all_graphs[0].pos, processed[0])
        coords_ro, feat_ro = find_nn_feat(coords[0], all_graphs[1].pos, processed[1])
        coords_lm, feat_lm = find_nn_feat(coords[1], all_graphs[2].pos, processed[2])
        coords_rm, feat_rm = find_nn_feat(coords[1], all_graphs[3].pos, processed[3])

        coords_orig = torch.cat((coords_lo, coords_ro), dim=-2)
        coords_mut = torch.cat((coords_lm, coords_rm), dim=-2)
        projected_orig = torch.cat((feat_lo, feat_ro), dim=-2)
        projected_mut = torch.cat((feat_lm, feat_rm), dim=-2)
        return coords_orig, coords_mut, projected_orig, projected_mut

    def aggregate(self, coords_orig, coords_mut, projected_orig, projected_mut):
        # Example coordinates and features
        orig_graph = create_pyg_graph_object(coords_orig, projected_orig)
        mut_graph = create_pyg_graph_object(coords_mut, projected_mut)

        orig_nodes = self.gcn(orig_graph)
        mut_nodes = self.gcn(mut_graph)

        # meanpool each graph and concatenate
        if self.use_max:
            orig_emb = torch.max(orig_nodes, dim=-2).values
            mut_emb = torch.max(mut_nodes, dim=-2).values
        elif self.use_mean:
            orig_emb = torch.mean(orig_nodes, dim=-2)
            mut_emb = torch.mean(mut_nodes, dim=-2)
        else:
            orig_emb = torch.sum(orig_nodes, dim=-2)
            mut_emb = torch.sum(mut_nodes, dim=-2)
        x = torch.cat((orig_emb, mut_emb), dim=-1)
        return x

    def forward(self, batch):
        """
        :param data:
        :return:
        """

        assert len(batch) == 1 or self.use_graph_only

        if not self.use_graph_only:
            data = batch[0]
            coords = data.coords
            # Unpack data
            if not self.use_graph_only:
                all_dict_feat = [unwrap_feats(geom_feat, device=self.device) for geom_feat in data.geom_feats]
                vertices = [dict_feat['vertices'] for dict_feat in all_dict_feat]
                if self.use_xyz:
                    # We need the vertices to push back the points.
                    # We also have to remove them from the dict to feed into base_nets
                    vertices1, coords0 = center_normalize(vertices[:2], [coords[0]])
                    vertices2, coords1 = center_normalize(vertices[2:], [coords[1]])
                    vertices = vertices1 + vertices2
                    coords = coords0 + coords1
                    for i, dict_feat in enumerate(all_dict_feat):
                        dict_feat["x_in"] = torch.cat([vertices[i], dict_feat["x_in"]], dim=1)
                    # TODO : align graphs
            if self.use_graph:
                all_graphs = [graph.to(self.device) for graph in data.graph_feats]

            # Run it through encoder
            if not self.use_graph:
                # We need to remove vertices here
                _ = [dict_feat.pop("vertices") for dict_feat in all_dict_feat]
                processed = [self.encoder_model(**dict_feat) for dict_feat in all_dict_feat]
            else:
                processed = [self.encoder_model(graph=graph, **dict_feat) for dict_feat, graph in
                             zip(all_dict_feat, all_graphs)]

            # Project it onto the coords
            coords_orig, coords_mut, projected_orig, projected_mut = self.project_processed_surface(vertices=vertices,
                                                                                                    processed=processed,
                                                                                                    coords=coords)
            x = self.aggregate(coords_orig, coords_mut, projected_orig, projected_mut)

            # Final MLP
            x = self.top_mlp(x)

        # GRAPH ONLY
        else:
            # Batch mut and orig graphs for efficient encoding.
            all_coords = [data.coords for data in batch]
            all_graphs_orig = list()
            all_graphs_mut = list()
            for data in batch:
                for graph in data.graph_feats[:2]:
                    graph.processed = torch.zeros(graph.num_nodes, 1)
                    all_graphs_orig.append(graph)
                for graph in data.graph_feats[2:]:
                    graph.processed = torch.zeros(graph.num_nodes, 1)
                    all_graphs_mut.append(graph)
            graphs_orig = Batch.from_data_list(all_graphs_orig)
            graphs_mut = Batch.from_data_list(all_graphs_mut)
            graphs_orig.processed = self.encoder_model(graphs_orig)
            graphs_mut.processed = self.encoder_model(graphs_mut)
            all_graphs_orig = graphs_orig.to_data_list()
            all_graphs_mut = graphs_mut.to_data_list()

            def group_consecutive_elements(lst):
                result = []
                for i in range(0, len(lst) - 1, 2):
                    result.append((lst[i], lst[i + 1]))
                return result

            all_graphs_orig = group_consecutive_elements(all_graphs_orig)
            all_graphs_mut = group_consecutive_elements(all_graphs_mut)

            all_embs = []
            for graph_orig, graph_mut, coords in zip(all_graphs_orig, all_graphs_mut, all_coords):
                glo, gro = graph_orig
                glm, grm = graph_mut
                all_graphs = [glo, gro, glm, grm]
                processed = [graph.processed for graph in all_graphs]
                coords_orig, coords_mut, projected_orig, projected_mut = self.project_processed_graph(
                    all_graphs=all_graphs, processed=processed, coords=coords)
                x = self.aggregate(coords_orig, coords_mut, projected_orig, projected_mut)
                # TODO BATCH FINAL GRAPH TOO ?
                x = self.top_mlp(x)
                all_embs.append(x)
            x = torch.stack(all_embs).view(-1)
            # x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=0.25, training=self.training)
            # x = self.fc2(x).view(-1)
        return x


def find_nn_feat(target, source, feat):
    dists = torch.cdist(target, source)
    min_indices = torch.argmin(dists, dim=1).unique()
    return source[min_indices], feat[min_indices]
