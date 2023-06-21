import torch

from atom2d_utils.learning_utils import unwrap_feats, center_normalize
import base_nets
from base_nets.layers import GCN, get_mlp, GraphDiffNet
from base_nets.utils import create_pyg_graph_object
from data_processing.point_cloud_utils import torch_rbf


class MSPSurfNet(torch.nn.Module):

    def __init__(self, in_channels=5, out_channel=64, C_width=128, N_block=4, hidden_sizes=(128,), drate=0.3,
                 batch_norm=False, use_max=True, use_xyz=False, use_graph=False, **kwargs):
        super(MSPSurfNet, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.use_max = use_max
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
            self.encoder_model = GraphDiffNet(C_in=in_channels,
                                              C_out=out_channel,
                                              C_width=C_width,
                                              N_block=N_block,
                                              last_activation=torch.relu)
        self.gcn = GCN(num_features=2 * (out_channel + 1), hidden_channels=out_channel, out_channel=out_channel,
                       drate=drate)
        self.top_mlp = get_mlp(in_features=2 * out_channel,
                               hidden_sizes=hidden_sizes,
                               drate=drate,
                               batch_norm=batch_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, coords):
        """
        Here inputs are [left_orig, right_orig, left_mut, right_mut] or
         ([left_orig, right_orig, left_mut, right_mut], graphs)

        :param x:
        :param coords: orig_coords, mut_coords
        :return:
        """

        if not self.use_graph:
            all_dict_feat = [unwrap_feats(geom_feat, device=self.device) for geom_feat in x]
            vertices = [dict_feat.pop('vertices') for dict_feat in all_dict_feat]
        else:
            all_dict_feat = [unwrap_feats(geom_feat, device=self.device) for geom_feat in x[0]]
            all_graphs = [graph.to(self.device) for graph in x[1]]
            vertices = [dict_feat['vertices'] for dict_feat in all_dict_feat]

        if self.use_xyz:
            vertices1, coords0 = center_normalize(vertices[:2], [coords[0]])
            vertices2, coords1 = center_normalize(vertices[2:], [coords[1]])
            vertices = vertices1 + vertices2
            coords = coords0 + coords1
            for i, dict_feat in enumerate(all_dict_feat):
                dict_feat["x_in"] = torch.cat([vertices[i], dict_feat["x_in"]], dim=1)

            # TODO : align graphs

        # We need the vertices to push back the points.
        # We also have to remove them from the dict to feed into base_nets
        if not self.use_graph:
            processed = [self.encoder_model(**dict_feat) for dict_feat in all_dict_feat]
        else:
            processed = [self.encoder_model(graph, **dict_feat) for dict_feat, graph in zip(all_dict_feat, all_graphs)]

        # Now project each part onto its points
        projected_left_orig = torch_rbf(points_1=vertices[0], feats_1=processed[0], points_2=coords[0], concat=True)
        projected_right_orig = torch_rbf(points_1=vertices[1], feats_1=processed[1], points_2=coords[0], concat=True)
        projected_left_mut = torch_rbf(points_1=vertices[2], feats_1=processed[2], points_2=coords[1], concat=True)
        projected_right_mut = torch_rbf(points_1=vertices[3], feats_1=processed[3], points_2=coords[1], concat=True)
        projected_orig = torch.cat((projected_left_orig, projected_right_orig), dim=-1)
        projected_mut = torch.cat((projected_left_mut, projected_right_mut), dim=-1)

        # Example coordinates and features
        orig_graph = create_pyg_graph_object(coords[0], projected_orig)
        mut_graph = create_pyg_graph_object(coords[1], projected_mut)

        orig_nodes = self.gcn(orig_graph)
        mut_nodes = self.gcn(mut_graph)

        # meanpool each graph and concatenate
        if self.use_max:
            orig_emb = torch.max(orig_nodes, dim=-2).values
            mut_emb = torch.max(mut_nodes, dim=-2).values
        else:
            orig_emb = torch.mean(orig_nodes, dim=-2)
            mut_emb = torch.mean(mut_nodes, dim=-2)
        x = torch.cat((orig_emb, mut_emb), dim=-1)

        x = self.top_mlp(x)
        # x = torch.sigmoid(x)
        return x
