import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

import base_nets
from .geometry import to_basis, from_basis


class LearnedTimeDiffusion(nn.Module):
    """
    Applies diffusion with learned per-channel t.
    In the spectral domain this becomes
        f_out = e ^ (lambda_i t) f_in
    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal
      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values
    """

    def __init__(self, C_inout, method="spectral"):
        super(LearnedTimeDiffusion, self).__init__()
        self.C_inout = C_inout
        self.diffusion_time = nn.Parameter(torch.Tensor(C_inout))  # (C)
        self.method = method  # one of ['spectral', 'implicit_dense']

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, x, L, mass, evals, evecs):

        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout
                )
            )

        if self.method == "spectral":

            # Transform to spectral
            x_spec = to_basis(x, evecs, mass)

            # Diffuse
            time = self.diffusion_time
            diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec

            # Transform back to per-vertex
            x_diffuse = from_basis(x_diffuse_spec, evecs)

        elif self.method == "implicit_dense":
            V = x.shape[-2]

            # Form the dense matrices (M + tL) with dims (B,C,V,V)
            mat_dense = L.to_dense().unsqueeze(1).expand(-1, self.C_inout, V, V).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)

            # Solve the system
            rhs = x * mass.unsqueeze(-1)
            rhsT = torch.transpose(rhs, 1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            x_diffuse = torch.transpose(sols.squeeze(-1), 1, 2)

        else:
            raise ValueError("unrecognized method")

        return x_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear layer to keep dimension down.

    Input:
        - vectors: (V,C,2)
    Output:
        - dots: (V,C) dots
    """

    def __init__(self, C_inout, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.C_inout = C_inout
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.C_inout, self.C_inout, bias=False)
            self.A_im = nn.Linear(self.C_inout, self.C_inout, bias=False)
        else:
            self.A = nn.Linear(self.C_inout, self.C_inout, bias=False)

        # self.norm = nn.InstanceNorm1d(C_inout)

    def forward(self, vectors):

        vectorsA = vectors  # (V,C)

        if self.with_gradient_rotations:
            vectorsBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vectorsBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vectorsBreal = self.A(vectors[..., 0])
            vectorsBimag = self.A(vectors[..., 1])

        dots = vectorsA[..., 0] * vectorsBreal + vectorsA[..., 1] * vectorsBimag

        return torch.tanh(dots)


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


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name="miniMLP"):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i), nn.Dropout(p=0.5)
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                ),
            )

            # Nonlinearity
            # (but not on the last layer)
            if not is_last:
                self.add_module(name + "_mlp_act_{:03d}".format(i), activation())


class DiffusionNetBlock(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(
            self,
            C_width,
            mlp_hidden_dims,
            dropout=True,
            diffusion_method="spectral",
            with_gradient_features=True,
            with_gradient_rotations=True,
    ):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method)

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                self.C_width, with_gradient_rotations=self.with_gradient_rotations
            )
            self.MLP_C += self.C_width

        # MLPs
        self.mlp = MiniMLP(
            [self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.dropout
        )

    def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):

        # Manage dimensions
        B = x_in.shape[0]  # batch dimension
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width
                )
            )

        # Diffusion block
        x_diffuse = self.diffusion(x_in, L, mass, evals, evecs)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = (
                []
            )  # Manually loop over the batch (if there is a batch dimension) since torch.mm() doesn't support batching
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b, ...], x_diffuse[b, ...])
                x_gradY = torch.mm(gradY[b, ...], x_diffuse[b, ...])

                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))
            x_grad = torch.stack(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features = self.gradient_features(x_grad)

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in, x_diffuse), dim=-1)

        # Apply the mlp
        x0_out = self.mlp(feature_combined)

        # Skip connection
        x0_out = x0_out + x_in

        return x0_out


class DiffusionNet(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            C_width=128,
            N_block=4,
            last_activation=None,
            outputs_at="vertices",
            mlp_hidden_dims=None,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
            diffusion_method="spectral",
    ):
        """
        Construct a DiffusionNet.
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

        super(DiffusionNet, self).__init__()

        # # Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ["vertices", "edges", "faces"]:
            raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ["spectral", "implicit_dense"]:
            raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # # Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(
                C_width=C_width,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
            )

            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(
            self,
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
        A forward pass on the DiffusionNet.
        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].
        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet,
        not all are strictly necessary.
        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
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

        # Apply the first linear layer
        x = self.first_lin(x_in)

        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x = self.last_lin(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == "vertices":
            x_out = x

        elif self.outputs_at == "edges":
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)

        elif self.outputs_at == "faces":
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out


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
            diffnet_block = base_nets.layers.DiffusionNetBlock(
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

    def forward(self, graph, *largs, **kwargs,):
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


class GraphNet(nn.Module):
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
