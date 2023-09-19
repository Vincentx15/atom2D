import torch
import torch.nn as nn
from torch_geometric.data import Batch

from base_nets import DiffusionNetBatch, GraphDiffNetParallel, GraphDiffNetSequential, GraphDiffNetAttention, \
    GraphDiffNetBipartite, AtomNetGraph, GCN
from base_nets.diffusion_net.layers import get_mlp
from base_nets.utils import create_pyg_graph_object
from data_processing.point_cloud_utils import torch_rbf


class MSPSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, hidden_sizes=(128,), drate=0.3,
                 batch_norm=False, use_max=True, use_mean=False, use_graph=False, use_graph_only=False,
                 output_graph=False, graph_model='parallel', use_gat=False, use_v2=False, neigh_th=8, flash=True, **kwargs):
        super(MSPSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.use_gat = use_gat
        self.use_max = use_max
        self.use_mean = use_mean
        self.output_graph = output_graph

        # Create the model
        self.use_graph = use_graph or use_graph_only
        self.use_graph_only = use_graph_only
        if use_graph_only:
            self.encoder_model = AtomNetGraph(C_in=in_channels,
                                              C_out=out_channel,
                                              C_width=C_width)
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
                                                          output_graph=output_graph)
            elif graph_model == 'sequential':
                self.encoder_model = GraphDiffNetSequential(C_in=in_channels,
                                                            C_out=out_channel,
                                                            C_width=C_width,
                                                            N_block=N_block,
                                                            last_activation=torch.relu,
                                                            use_bn=batch_norm,
                                                            output_graph=output_graph)
            elif graph_model == 'attention':
                self.encoder_model = GraphDiffNetAttention(C_in=in_channels,
                                                           C_out=out_channel,
                                                           C_width=C_width,
                                                           N_block=N_block,
                                                           last_activation=torch.relu,
                                                           use_bn=batch_norm,
                                                           output_graph=output_graph,
                                                           flash=flash,
                                                           )
            elif graph_model == 'bipartite':
                self.encoder_model = GraphDiffNetBipartite(C_in=in_channels,
                                                           C_out=out_channel,
                                                           C_width=C_width,
                                                           N_block=N_block,
                                                           last_activation=torch.relu,
                                                           use_bn=batch_norm,
                                                           output_graph=output_graph,
                                                           use_gat=use_gat,
                                                           use_v2=use_v2,
                                                           neigh_th=neigh_th)
        if self.use_graph_only:
            # Follow atom3D
            infeature_gcn = 2 * (out_channel + 1) if not self.use_graph_only else C_width * 2
        elif self.output_graph:
            # Just take graph output
            infeature_gcn = out_channel
        else:
            # Concatenate with torch rbf in a weird way
            infeature_gcn = 2 * (out_channel + 1)

        self.gcn = GCN(num_features=infeature_gcn, hidden_channels=out_channel, out_channel=out_channel,
                       drate=drate)
        self.top_net_graph = nn.Sequential(*[
            nn.Linear(out_channel * 2, out_channel * 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(out_channel * 2, 1)
        ])

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
        graph_lo, graph_ro, graph_lm, graph_rm = None, None, None, None
        surface_lo, surface_ro, surface_lm, surface_rm = None, None, None, None
        coords = batch.coords
        if not self.use_graph_only:
            surface_lo, surface_ro, surface_lm, surface_rm = batch.surface_lo, batch.surface_ro, batch.surface_lm, batch.surface_rm
            verts_lo, verts_ro, verts_lm, verts_rm = surface_lo.vertices, surface_ro.vertices, surface_lm.vertices, surface_rm.vertices
            vertices = [[x, y, z, w] for x, y, z, w in zip(verts_lo, verts_ro, verts_lm, verts_rm)]
        if self.use_graph:
            graph_lo, graph_ro, graph_lm, graph_rm = batch.graph_lo, batch.graph_ro, batch.graph_lm, batch.graph_rm
            graphs = [[x, y, z, w] for x, y, z, w in zip(graph_lo.to_data_list(), graph_ro.to_data_list(),
                                                         graph_lm.to_data_list(), graph_rm.to_data_list())]
        if self.use_graph_only:
            all_graphs_orig = [[x, y] for x, y in zip(graph_lo.to_data_list(), graph_ro.to_data_list())]
            all_graphs_mut = [[x, y] for x, y in zip(graph_lm.to_data_list(), graph_rm.to_data_list())]
            all_graphs_orig = Batch.from_data_list([y for x in all_graphs_orig for y in x])
            all_graphs_mut = Batch.from_data_list([y for x in all_graphs_mut for y in x])
            processed_orig = self.encoder_model(graph=all_graphs_orig).split(all_graphs_orig.batch.bincount().tolist())
            processed_mut = self.encoder_model(graph=all_graphs_mut).split(all_graphs_mut.batch.bincount().tolist())
            proc_lo = [processed_orig[i] for i in range(0, len(processed_orig), 2)]
            proc_ro = [processed_orig[i] for i in range(1, len(processed_orig), 2)]
            proc_lm = [processed_mut[i] for i in range(0, len(processed_mut), 2)]
            proc_rm = [processed_mut[i] for i in range(1, len(processed_mut), 2)]
            processed = [[x, y, z, w] for x, y, z, w in zip(proc_lo, proc_ro, proc_lm, proc_rm)]
        else:
            processed_lo = self.encoder_model(graph=graph_lo, surface=surface_lo)
            processed_ro = self.encoder_model(graph=graph_ro, surface=surface_ro)
            processed_lm = self.encoder_model(graph=graph_lm, surface=surface_lm)
            processed_rm = self.encoder_model(graph=graph_rm, surface=surface_rm)
            processed = [[x, y, z, w] for x, y, z, w in zip(processed_lo, processed_ro, processed_lm, processed_rm)]

        xs = []
        if self.use_graph_only or (self.use_graph and self.output_graph):
            for graph, proc, coord in zip(graphs, processed, coords):
                coords_orig, coords_mut, projected_orig, projected_mut = self.project_processed_graph(all_graphs=graph,
                                                                                                      processed=proc,
                                                                                                      coords=coord)
                x = self.aggregate(coords_orig, coords_mut, projected_orig, projected_mut)
                x = self.top_net_graph(x)
                xs.append(x)
        else:
            for verts, proc, coord in zip(vertices, processed, coords):
                coords_orig, coords_mut, projected_orig, projected_mut = self.project_processed_surface(vertices=verts,
                                                                                                        processed=proc,
                                                                                                        coords=coord)
                x = self.aggregate(coords_orig, coords_mut, projected_orig, projected_mut)
                x = self.top_net_graph(x)
                xs.append(x)

        x = torch.stack(xs, dim=0).flatten()
        return x


def find_nn_feat(target, source, feat):
    dists = torch.cdist(target, source)
    min_indices = torch.argmin(dists, dim=1).unique()
    return source[min_indices], feat[min_indices]
