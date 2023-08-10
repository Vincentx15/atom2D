from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from base_nets import DiffusionNetBlock


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
        self.first_lin1 = nn.Linear(C_in, diffnet_width)
        self.first_lin2 = nn.Linear(C_in, diffnet_width)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlock(
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

    def forward(self, graph=None, surface=None, x_in=None, mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None, faces=None):
        """
        A forward pass on the MixedNet.
        """
        x_in, mass, L, evals, evecs, gradX, gradY = surface.x, surface.mass, surface.L, surface.evals, surface.evecs, surface.gradX, surface.gradY
        vertices = surface.vertices
        mass = [m.unsqueeze(0) for m in mass]
        L = [ll.unsqueeze(0) for ll in L]
        evals = [e.unsqueeze(0) for e in evals]
        evecs = [e.unsqueeze(0) for e in evecs]

        # Precompute distance
        sigma = 2.5
        with torch.no_grad():
            all_dists = torch.cdist(vertices, graph.pos)
            rbf_weights = torch.exp(-all_dists / sigma)

        # Apply the first linear layer
        diff_x = self.first_lin1(x_in)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        # todo update communication with batch
        for graph_block, diff_block, mixer_block in zip(self.gcn_blocks, self.diff_blocks, self.mixer_blocks):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = graph_block(graph)
            # not necessary, cdist is fast
            # diff_on_graph = torch.sparse.mm(rbf_graph_surf, diff_x[0])
            # graph_on_diff = torch.sparse.mm(rbf_surf_graph, graph_x)
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

        return x_out


class GraphDiffNetSequential(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, dropout=True,
                 with_gradient_features=True,
                 with_gradient_rotations=True, diffusion_method="spectral"):
        """
        Construct a MixedNet in a sequential manner, with DiffusionNet blocks followed by GCN blocks.
        instead of the // architecture GraphDiffNet
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

        super().__init__()

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
        self.first_lin1 = nn.Linear(C_in, diffnet_width)
        self.first_lin2 = nn.Linear(C_in, diffnet_width)
        self.last_lin = nn.Linear(diffnet_width, C_out)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlock(
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

    def forward(self, graph, vertices, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None,
                faces=None):
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
        diff_x = self.first_lin1(x_in)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        for graph_block, diff_block in zip(self.gcn_blocks, self.diff_blocks):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = torch.mm(rbf_weights.T, diff_x[0])
            graph.x = graph_block(graph)
            diff_x = torch.mm(rbf_weights, graph.x)[None, ...]

        # Apply the last linear layer
        diff_x = self.last_lin(diff_x)

        x_out = diff_x
        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


class GraphDiffNetAttention(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, dropout=True,
                 with_gradient_features=True,
                 with_gradient_rotations=True,
                 diffusion_method="spectral"
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

        super().__init__()

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
        self.first_lin1 = nn.Linear(C_in, diffnet_width)
        self.first_lin2 = nn.Linear(C_in, diffnet_width)
        self.last_lin = nn.Linear(diffnet_width, C_out)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlock(
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

        self.att_graph_blocks = []
        for i_block in range(self.N_block):
            att_block = AttentionalPropagation(diffnet_width, num_heads=4)
            self.att_graph_blocks.append(att_block)
            self.add_module("att_graph_blocks_" + str(i_block), att_block)

        self.att_diff_blocks = []
        for i_block in range(self.N_block):
            att_block = AttentionalPropagation(diffnet_width, num_heads=4)
            self.att_diff_blocks.append(att_block)
            self.add_module("att_diff_blocks_" + str(i_block), att_block)

    def forward(self, graph, vertices, x_in, mass, L=None, evals=None, evecs=None, gradX=None, gradY=None, edges=None,
                faces=None):
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

        # Precompute distance (not used for now)
        # sigma = 2.5
        # with torch.no_grad():
        #     all_dists = torch.cdist(vertices, graph.pos)
        #     rbf_weights = torch.exp(-all_dists / sigma)

        # Apply the first linear layer
        diff_x = self.first_lin1(x_in)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        for graph_block, diff_block, att_graph_block, att_diff_block in zip(self.gcn_blocks, self.diff_blocks,
                                                                            self.att_graph_blocks,
                                                                            self.att_diff_blocks):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = graph_block(graph)
            diff_x = att_diff_block(diff_x, graph.x)
            graph.x = att_graph_block(graph.x, diff_x)

        # Apply the last linear layer
        diff_x = self.last_lin(diff_x)

        x_out = diff_x
        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


class GraphDiffNetBipartite(nn.Module):
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
        Channels are split into graphs and diff_block channels, then convoluted, then mixed using GCN
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

        super(GraphDiffNetBipartite, self).__init__()

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
        self.first_lin1 = nn.Linear(C_in, diffnet_width)
        self.first_lin2 = nn.Linear(C_in, diffnet_width)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlock(
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

        self.graphsurf_blocks = []
        for i_block in range(self.N_block):
            graphsurf_block = GCNConv(diffnet_width,
                                      diffnet_width if i_block < self.N_block - 1 else C_out,
                                      # add_self_loops=False,
                                      )
            self.graphsurf_blocks.append(graphsurf_block)
            self.add_module("graphsurf_block_" + str(i_block), graphsurf_block)

        self.surfgraph_blocks = []
        for i_block in range(self.N_block):
            surfgraph_block = GCNConv(diffnet_width,
                                      diffnet_width if i_block < self.N_block - 1 else C_out,
                                      # add_self_loops=False,
                                      )
            self.surfgraph_blocks.append(surfgraph_block)
            self.add_module("surfgraph_block_" + str(i_block), surfgraph_block)

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
            # rbf_surf_graph=None,
            # rbf_graph_surf=None,
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

        # Precompute bipartite graph
        sigma = 4
        with torch.no_grad():
            all_dists = torch.cdist(vertices, graph.pos)
            neighbors = torch.where(all_dists < 8)
            # Slicing requires tuple
            dists = all_dists[neighbors]
            dists = torch.exp(-dists / sigma)
            neighbors = torch.stack(neighbors).long()
            neighbors[1] += len(vertices)
            reverse_neighbors = torch.flip(neighbors, dims=(0,))
            all_pos = torch.cat((vertices, graph.pos))
            bipartite_surfgraph = Data(all_pos=all_pos, edge_index=neighbors, edge_weight=dists)
            bipartite_graphsurf = Data(all_pos=all_pos, edge_index=reverse_neighbors, edge_weight=dists)

        # Apply the first linear layer
        diff_x = self.first_lin1(x_in)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        for graph_block, diff_block, graphsurf_block, surfgraph_block in zip(self.gcn_blocks,
                                                                             self.diff_blocks,
                                                                             self.graphsurf_blocks,
                                                                             self.surfgraph_blocks):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = graph_block(graph)

            # Now use this for message passing. We can't use self loop with two GCN so average for now
            # (maybe mixer later ?)
            input_feats = torch.cat((diff_x[0], graph.x))
            out_surf = graphsurf_block(input_feats,
                                       bipartite_graphsurf.edge_index,
                                       bipartite_graphsurf.edge_weight)

            # The connectivity is well taken into account, but removing self loop yields zero outputs
            # TODO: debug
            # out_surf_test = graphsurf_block(input_feats,
            #                                 bipartite_graphsurf.edge_index[:, :-5],
            #                                 bipartite_graphsurf.edge_weight[:-5])

            out_graph = surfgraph_block(input_feats,
                                        bipartite_surfgraph.edge_index,
                                        bipartite_surfgraph.edge_weight)
            output_feat = torch.stack((out_surf, out_graph), dim=0)
            output_feat = torch.mean(output_feat, dim=0)
            diff_x = output_feat[:len(vertices)][None, ...]
            graph.x = output_feat[len(vertices):]

        x_out = diff_x
        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


class AtomNetGraph(torch.nn.Module):
    def __init__(self, C_in, C_out, C_width, last_factor=2):
        super().__init__()

        self.conv1 = GCNConv(C_in, C_width)
        self.bn1 = nn.BatchNorm1d(C_width)
        self.conv2 = GCNConv(C_width, C_width * 2)
        self.bn2 = nn.BatchNorm1d(C_width * 2)
        self.conv3 = GCNConv(C_width * 2, C_width * 4)
        self.bn3 = nn.BatchNorm1d(C_width * 4)
        self.conv4 = GCNConv(C_width * 4, C_width * 4)
        self.bn4 = nn.BatchNorm1d(C_width * 4)
        self.conv5 = GCNConv(C_width * 4, C_width * last_factor)
        self.bn5 = nn.BatchNorm1d(C_width * last_factor)

    def forward(self, graph, *largs, **kwargs, ):
        x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)
        return x


class GraphNet(nn.Module):  # deprecated
    def __init__(
            self,
            C_in,
            C_out,
            C_width=128,
            N_block=4,
            last_activation=None,
            dropout=True,
    ):
        """
        Construct a GraphNet.
        Parameters:
            C_in (int):                     input dimension
            C_out (int):                    output dimension
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces'].
            (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal graph blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
        """

        super().__init__()

        # # Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.dropout = True

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        self.gcn_blocks = []
        for i_block in range(self.N_block):
            gcn_block = GCN(C_width, C_width, C_width, drate=0.5 if dropout else 0, )
            self.gcn_blocks.append(gcn_block)
            self.add_module("gcn_block_" + str(i_block), gcn_block)

    def forward(
            self,
            graph,
            vertices,
            *largs,
            **kwargs,
            # rbf_surf_graph=None,
            # rbf_graph_surf=None,
    ):
        """
        A forward pass on the MixedNet.
        """

        # Precompute distance
        sigma = 2.5
        with torch.no_grad():
            all_dists = torch.cdist(vertices, graph.pos)
            rbf_weights = torch.exp(-all_dists / sigma)

        # Apply the first linear layer
        graph.x = self.first_lin(graph.x)

        # Apply each of the blocks
        for graph_block in self.gcn_blocks:
            graph.x = graph_block(graph)

        # Apply the last linear layer
        graph.x = self.last_lin(graph.x)

        # project features to surface
        x_out = torch.mm(rbf_weights, graph.x)

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        return x_out


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    torch.cuda.empty_cache()
    prob = torch.nn.functional.softmax(scores, dim=-1)
    torch.cuda.empty_cache()
    result = torch.einsum("bhnm,bdhm->bdhn", prob, value)
    torch.cuda.empty_cache()
    return result, prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [ll(x).view(batch_dim, self.dim, self.num_heads, -1) for ll, x in
                             zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        add_dim = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            add_dim = True
        if source.ndim == 2:
            source = source.unsqueeze(0)
        x, source = x.transpose(1, 2), source.transpose(1, 2)

        message = self.attn(x, source, source)
        x = self.mlp(torch.cat([x, message], dim=1)).transpose(1, 2)
        if add_dim:
            x = x.squeeze(0)
        return x
