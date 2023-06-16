import diff_net
import torch
import torch.nn as nn

from atom2d_utils.learning_utils import unwrap_feats, center_normalize
from data_processing.point_cloud_utils import torch_rbf

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


def create_pyg_graph_object(coords, features, sigma=3):
    num_nodes = len(coords)
    device = coords.device

    # Calculate pairwise distances using torch.cdist
    with torch.no_grad():
        pairwise_distances = torch.cdist(coords, coords)
        rbf_weights = torch.exp(-pairwise_distances / sigma)

        # Create edge index using torch.triu_indices and remove self-loops
        row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
        edge_index = torch.stack([row, col], dim=0)

        # Create bidirectional edges by concatenating (row, col) and (col, row)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

        # Extract edge weights from pairwise_distances using the created edge_index
        edge_weight = rbf_weights[row, col]
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0).to(device)

    return Data(x=features, edge_index=edge_index, edge_weight=edge_weight)


# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channel, drate=None):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channel)
        self.drate = drate

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        if self.drate is not None:
            x = F.dropout(x, p=self.drate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def get_mlp(in_features, hidden_sizes, batch_norm=True, drate=None):
    layers = []
    for units in hidden_sizes:
        layers.extend([
            nn.Linear(in_features, units),
            nn.ReLU()
        ])
        if batch_norm:
            layers.append(nn.BatchNorm1d(units))
        if drate is not None:
            layers.append(nn.Dropout(drate))
        in_features = units

    # Final FC layer
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


class GraphDiffNet(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            C_width=128,
            N_block=4,
            last_activation=None,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
            diffusion_method="spectral",
    ):
        """
        Construct a MixedNet.
        Channels are split into graphs and diff_block channels, then convoluted, then mixed
        Parameters:
            C_in (int):                     input dimension
            C_out (int):                    output dimension
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces'].
            (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0,
            saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient.
            Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(GraphDiffNet, self).__init__()

        # # Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.dropout = True

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ["spectral", "implicit_dense"]:
            raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # # Set up the network
        # channels are split into graphs and diff_block channels, then convoluted, then mixed
        diffnet_width = C_width // 2

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, diffnet_width)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = diff_net.layers.DiffusionNetBlock(
                C_width=diffnet_width,
                mlp_hidden_dims=self.mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
            )

            self.diff_blocks.append(diffnet_block)
            self.add_module("diffnet_block_" + str(i_block), self.diff_blocks[-1])

        self.gcn_blocks = []
        for i_block in range(self.N_block):
            gcn_block = GCN(diffnet_width, diffnet_width, diffnet_width, drate=0.5 if dropout else 0, )
            self.gcn_blocks.append(gcn_block)
            self.add_module("gcn_block_" + str(i_block), gcn_block)

        self.mixer_blocks = []
        for i_block in range(self.N_block):
            mixer_block = nn.Linear(diffnet_width * 2, diffnet_width if i_block < self.N_block - 1 else C_out)
            self.mixer_blocks.append(mixer_block)
            self.add_module("mixer_" + str(i_block), mixer_block)

    def forward(
            self,
            graph,
            vertices,
            x_in,
            mass,
            L=None,
            evals=None,
            evecs=None,
            gradX=None,
            gradY=None,
            edges=None,
            faces=None,
    ):
        """
        A forward pass on the MixedNet.
        """

        # # Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in:
            raise ValueError(
                "DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(
                    self.C_in, x_in.shape[-1]
                )
            )
        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            if L is not None:
                L = L.unsqueeze(0)
            if evals is not None:
                evals = evals.unsqueeze(0)
            if evecs is not None:
                evecs = evecs.unsqueeze(0)
            if gradX is not None:
                gradX = gradX.unsqueeze(0)
            if gradY is not None:
                gradY = gradY.unsqueeze(0)
            if edges is not None:
                edges = edges.unsqueeze(0)
            if faces is not None:
                faces = faces.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False

        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        # Precompute distance
        sigma = 2.5
        with torch.no_grad():
            all_dists = torch.cdist(vertices, graph.pos)
            rbf_weights = torch.exp(-all_dists / sigma)

        # Apply the first linear layer
        diff_x = self.first_lin(x_in)
        graph.x = self.first_lin(graph.x)

        # Apply each of the blocks
        for graph_block, diff_block, mixer_block in zip(self.gcn_blocks, self.diff_blocks, self.mixer_blocks):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = graph_block(graph)
            diff_on_graph = torch.mm(rbf_weights.T, diff_x[0])
            graph_on_diff = torch.mm(rbf_weights, graph.x)
            cat_graph = torch.cat((diff_on_graph, graph.x), dim=1)  # TODO : two mixers ? sequential model?
            cat_diff = torch.cat((diff_x, graph_on_diff[None, ...]), dim=2)
            graph.x = mixer_block(cat_graph)
            diff_x = mixer_block(cat_diff)

        x_out = diff_x
        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


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
            self.encoder_model = diff_net.layers.DiffusionNet(C_in=in_channels,
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
        # We also have to remove them from the dict to feed into diff_net
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
