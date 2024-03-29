from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

try:
    from flash_attn import flash_attn_func
except ImportError:
    pass

from base_nets import DiffusionNetBlockBatch


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channel, drate=None, use_bn=False, use_distance=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channel)
        self.use_bn = use_bn
        self.use_distance = use_distance
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(out_channel)
        self.drate = drate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if self.use_distance else None
        x = self.conv1(x, edge_index, edge_weight)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        if self.drate is not None:
            x = F.dropout(x, p=self.drate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if self.use_bn:
            x = self.bn2(x)
        return x


class WLNConvLast(MessagePassing):

    def __init__(self, hsize: int, bias: bool):
        super(WLNConvLast, self).__init__(aggr='mean')
        self.hsize = hsize
        self.bias = bias
        self._build_components()

    def _build_components(self):
        self.W0 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.W2 = nn.Linear(self.hsize, self.hsize, self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        mess = self.W0(x_i) * self.W1(edge_attr) * self.W2(x_j)
        return mess


class WLNConv(MessagePassing):
    def __init__(self,
                 node_fdim: int,
                 edge_fdim: int,
                 depth: int,
                 hsize: int,
                 bias: bool = False,
                 dropout: float = 0.2,
                 activation: str = 'relu'):
        super(WLNConv, self).__init__(
            aggr='mean')  # We use mean here because the node embeddings started to explode otherwise
        self.hsize = hsize
        self.bias = bias
        self.depth = depth
        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.dropout_p = dropout
        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'lrelu':
            self.activation_fn = F.leaky_relu
        self._build_components()

    def _build_components(self):
        self.node_emb = nn.Linear(self.node_fdim, self.hsize, self.bias)
        self.mess_emb = nn.Linear(self.edge_fdim, self.hsize, self.bias)
        self.U1 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.U2 = nn.Linear(self.hsize, self.hsize, self.bias)
        self.V = nn.Linear(2 * self.hsize, self.hsize, self.bias)

        self.dropouts = []
        self.bns = []
        self.bns_2 = []
        for i in range(self.depth):
            self.dropouts.append(nn.Dropout(p=self.dropout_p))
            self.bns.append(nn.BatchNorm1d(self.hsize))
            self.bns_2.append(nn.BatchNorm1d(self.hsize))
        self.dropouts = nn.ModuleList(self.dropouts)
        self.bns = nn.ModuleList(self.bns)
        self.bns_2 = nn.ModuleList(self.bns_2)
        self.conv_last = WLNConvLast(hsize=self.hsize, bias=self.bias)

    def forward(self, graph):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        if x.size(-1) != self.hsize:
            x = self.node_emb(x)
        edge_attr = self.mess_emb(edge_attr)

        x_depths = []
        for i in range(self.depth):
            x = self.dropouts[i](x)
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            x = self.bns[i](x)
            x_depth = self.conv_last(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns_2[i](x)
            x_depths.append(x_depth)
        x_final = x_depths[-1]
        return x_final

    def update(self, inputs, x):
        x = self.activation_fn(self.U1(x) + self.U2(inputs))
        return x

    def message(self, x_j, edge_attr):
        nei_mess = self.activation_fn(self.V(torch.cat([x_j, edge_attr], dim=-1)))
        return nei_mess


class AddAggregate(MessagePassing):
    def __init__(self, aggr='add'):
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index, edge_weights):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights)
        return out

    def message(self, x_j, edge_weights):
        # x_j has shape [E, out_channels]
        return edge_weights[..., None] * x_j


def compute_bipartite_graphs(vertices, graph, neigh_th):
    sigma = neigh_th / 2  # todo is this the right way to do it?
    with torch.no_grad():
        all_dists = [torch.cdist(vert, mini_graph.pos) for vert, mini_graph in zip(vertices, graph.to_data_list())]
        neighbors = [torch.where(x < neigh_th) for x in all_dists]
        # Slicing requires tuple
        dists = [all_dist[neigh] for all_dist, neigh in zip(all_dists, neighbors)]
        dists = [torch.exp(-x / sigma) for x in dists]
        neighbors = [torch.stack(x).long() for x in neighbors]
        for i, neighbor in enumerate(neighbors):
            neighbor[1] += len(vertices[i])
        reverse_neighbors = [torch.flip(neigh, dims=(0,)) for neigh in neighbors]
        all_pos = [torch.cat((vert, mini_graph.pos)) for vert, mini_graph in zip(vertices, graph.to_data_list())]
        bipartite_surfgraph = [Data(all_pos=pos, edge_index=neighbor, edge_weight=dist) for pos, neighbor, dist in
                               zip(all_pos, neighbors, dists)]
        bipartite_graphsurf = [Data(all_pos=pos, edge_index=rneighbor, edge_weight=dist) for pos, rneighbor, dist in
                               zip(all_pos, reverse_neighbors, dists)]
        return bipartite_graphsurf, bipartite_surfgraph


class GraphDiffNetParallel(nn.Module):
    def __init__(
            self,
            C_in,
            C_out,
            C_width=128,
            N_block=4,
            last_activation=None,
            dropout=0.5,
            with_gradient_features=True,
            with_gradient_rotations=True,
            diffusion_method="spectral",
            use_bn=True,
            use_mp=False,
            neigh_thresh=8,
            output_graph=False,
            use_distance=False
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
            dropout (float):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0,
            saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient.
            Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(GraphDiffNetParallel, self).__init__()

        # # Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.output_graph = output_graph

        # Outputs
        self.last_activation = last_activation
        self.dropout = dropout

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
        self.first_lin1 = nn.Linear(5, diffnet_width)
        self.first_lin2 = nn.Linear(C_in, diffnet_width)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlockBatch(
                C_width=diffnet_width,
                mlp_hidden_dims=self.mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                use_bn=use_bn
            )

            self.diff_blocks.append(diffnet_block)
            self.add_module("diffnet_block_" + str(i_block), self.diff_blocks[-1])

        self.gcn_blocks = []
        for i_block in range(self.N_block):
            gcn_block = GCN(diffnet_width, diffnet_width, diffnet_width,
                            drate=dropout, use_bn=use_bn, use_distance=use_distance)
            self.gcn_blocks.append(gcn_block)
            self.add_module("gcn_block_" + str(i_block), gcn_block)

        self.neigh_thresh = neigh_thresh
        self.use_mp = use_mp
        if use_mp:
            self.mp = AddAggregate()

        self.mixer_blocks = []
        for i_block in range(self.N_block):
            mixer_block = nn.Linear(diffnet_width * 2, diffnet_width if i_block < self.N_block - 1 else C_out)
            self.mixer_blocks.append(mixer_block)
            self.add_module("mixer_" + str(i_block), mixer_block)

    def forward(self, graph=None, surface=None):
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
        if self.use_mp:
            bipartite_graphsurf, bipartite_surfgraph = compute_bipartite_graphs(vertices, graph,
                                                                                neigh_th=self.neigh_thresh)
        else:
            sigma = 2.5
            with torch.no_grad():
                all_dists = [torch.cdist(vert, mini_graph.pos) for vert, mini_graph in
                             zip(vertices, graph.to_data_list())]
                rbf_weights = [torch.exp(-x / sigma) for x in all_dists]

        # Apply the first linear layer
        split_sizes = [tensor.size(0) for tensor in x_in]
        diff_x = torch.split(self.first_lin1(torch.cat(x_in, dim=0)), split_sizes, dim=0)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        for graph_block, diff_block, mixer_block in zip(self.gcn_blocks,
                                                        self.diff_blocks,
                                                        self.mixer_blocks):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = graph_block(graph)
            if self.use_mp:
                input_feats = [torch.cat((diff, mini_graph.x)) for diff, mini_graph in
                               zip(diff_x, graph.to_data_list())]
                diff_on_graph = [self.mp(input_feat, bisurfgraph.edge_index, bisurfgraph.edge_weight)
                                 for input_feat, bisurfgraph in zip(input_feats, bipartite_surfgraph)]
                graph_on_diff = [self.mp(input_feat, bigraphsurf.edge_index, bigraphsurf.edge_weight)
                                 for input_feat, bigraphsurf in zip(input_feats, bipartite_graphsurf)]
                graph_on_diff = [out[:len(vert)] for out, vert in zip(graph_on_diff, vertices)]
                diff_on_graph = [out[len(vert):] for out, vert in zip(diff_on_graph, vertices)]
            else:
                diff_on_graph = [torch.mm(rbf_w.T, x) for rbf_w, x in zip(rbf_weights, diff_x)]
                graph_on_diff = [torch.mm(rbf_w, mini_graph.x) for rbf_w, mini_graph in
                                 zip(rbf_weights, graph.to_data_list())]
            cat_graph = [torch.cat((diff_on_graph[i], mini_graph.x), dim=1) for i, mini_graph in
                         enumerate(graph.to_data_list())]
            cat_diff = [torch.cat((diff_x[i], graph_on_diff[i]), dim=1) for i in range(len(diff_x))]
            diff_x = torch.split(mixer_block(torch.cat(cat_diff, dim=0)), split_sizes, dim=0)
            graph.x = mixer_block(
                torch.cat(cat_graph, dim=0))  # todo check if this is correct (or should we use .from_data_list)

        if self.output_graph:
            x_out = [mini_graph.x for mini_graph in graph.to_data_list()]
        else:
            x_out = diff_x
        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = [self.last_activation(x) for x in x_out]

        return x_out


class GraphDiffNetSequential(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, dropout=0.5,
                 with_gradient_features=True, with_gradient_rotations=True, diffusion_method="spectral", use_bn=True,
                 output_graph=False, use_mp=False, use_gat=False, use_skip=False, neigh_thresh=8, use_distance=False):
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
            dropout (float):                 if True, internal MLPs use dropout (default: True)
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
        self.output_graph = output_graph

        self.neigh_thresh = neigh_thresh

        # Outputs
        self.last_activation = last_activation
        self.dropout = dropout

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
        self.first_lin1 = nn.Linear(5, diffnet_width)
        self.first_lin2 = nn.Linear(C_in, diffnet_width)
        self.last_lin = nn.Linear(diffnet_width, C_out)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlockBatch(
                C_width=diffnet_width,
                mlp_hidden_dims=self.mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                use_bn=use_bn
            )

            self.diff_blocks.append(diffnet_block)
            self.add_module("diffnet_block_" + str(i_block), self.diff_blocks[-1])

        self.gcn_blocks = []
        for i_block in range(self.N_block):
            gcn_block = GCN(diffnet_width, diffnet_width, diffnet_width,
                            drate=dropout, use_bn=use_bn, use_distance=use_distance)
            self.gcn_blocks.append(gcn_block)
            self.add_module("gcn_block_" + str(i_block), gcn_block)

        self.use_mp = use_mp
        self.use_skip = use_skip
        if use_mp:
            conv_layer = GCNConv if not use_gat else GATConv
            self.graphsurf_blocks = []
            for i_block in range(self.N_block):
                graphsurf_block = conv_layer(diffnet_width,
                                             diffnet_width,
                                             add_self_loops=True,
                                             )
                self.graphsurf_blocks.append(graphsurf_block)
                self.add_module("graphsurf_block_" + str(i_block), graphsurf_block)

            self.surfgraph_blocks = []
            for i_block in range(self.N_block):
                surfgraph_block = conv_layer(diffnet_width,
                                             diffnet_width,
                                             add_self_loops=True,
                                             )
                self.surfgraph_blocks.append(surfgraph_block)
                self.add_module("surfgraph_block_" + str(i_block), surfgraph_block)

    def forward(self, graph=None, surface=None):
        """
        A forward pass on the MixedNet.
        """

        x_in, mass, L, evals, evecs, gradX, gradY = surface.x, surface.mass, surface.L, surface.evals, surface.evecs, surface.gradX, surface.gradY
        vertices = surface.vertices
        mass = [m.unsqueeze(0) for m in mass]
        L = [ll.unsqueeze(0) for ll in L]
        evals = [e.unsqueeze(0) for e in evals]
        evecs = [e.unsqueeze(0) for e in evecs]

        if self.use_mp:
            bipartite_graphsurf, bipartite_surfgraph = compute_bipartite_graphs(vertices, graph,
                                                                                neigh_th=self.neigh_thresh)
        else:  # Precompute distance
            sigma = 2.5
            with torch.no_grad():
                all_dists = [torch.cdist(vert, mini_graph.pos) for vert, mini_graph in
                             zip(vertices, graph.to_data_list())]
                rbf_weights = [torch.exp(-x / sigma) for x in all_dists]

        # Apply the first linear layer
        split_sizes = [tensor.size(0) for tensor in x_in]
        diff_x = torch.split(self.first_lin1(torch.cat(x_in, dim=0)), split_sizes, dim=0)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        NL = len(self.gcn_blocks)
        for i, (graph_block, diff_block) in enumerate(zip(self.gcn_blocks, self.diff_blocks)):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            if self.use_mp:
                # Now use this for message passing.
                input_feats = [torch.cat((diff, mini_graph.x)) for diff, mini_graph in
                               zip(diff_x, graph.to_data_list())]
                surfgraph_block = self.surfgraph_blocks[i]
                out_graph = [surfgraph_block(input_feat, bisurfgraph.edge_index, bisurfgraph.edge_weight)
                             for input_feat, bisurfgraph in zip(input_feats, bipartite_surfgraph)]
                graph_x_out = torch.cat([out[len(vert):] for out, vert in zip(out_graph, vertices)], dim=0)
                graph.x = graph.x + graph_x_out if (self.use_skip and i < NL) else graph_x_out
            else:
                graph.x = torch.cat([torch.mm(rbf_w.T, x) for rbf_w, x in zip(rbf_weights, diff_x)], dim=0)
            graph.x = graph_block(graph)
            if self.use_mp:
                # Now use this for message passing.
                input_feats = [torch.cat((diff, mini_graph.x)) for diff, mini_graph in
                               zip(diff_x, graph.to_data_list())]
                graphsurf_block = self.graphsurf_blocks[i]
                out_surf = [graphsurf_block(input_feat, bigraphsurf.edge_index, bigraphsurf.edge_weight)
                            for input_feat, bigraphsurf in zip(input_feats, bipartite_graphsurf)]
                diff_x_out = [out[:len(vert)] for out, vert in zip(out_surf, vertices)]
                diff_x = [x + y for x, y in zip(diff_x, diff_x_out)] if (self.use_skip and i < NL) else diff_x_out
            else:
                diff_x = [torch.mm(rbf_w, mini_graph.x) for rbf_w, mini_graph in zip(rbf_weights, graph.to_data_list())]

        if self.output_graph:
            x_out = [mini_graph.x for mini_graph in graph.to_data_list()]
        else:
            x_out = diff_x
        # Apply last linear :
        x_out = [self.last_lin(x) for x in x_out]

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = [self.last_activation(x) for x in x_out]

        return x_out


class GraphDiffNetBipartite(nn.Module):
    def __init__(self, C_in_graph, C_out, C_in_surf=5, C_width=128, N_block=4, last_activation=None, dropout=0.5,
                 with_gradient_features=True, with_gradient_rotations=True, diffusion_method="spectral", use_bn=True,
                 output_graph=False, use_gat=False, use_v2=False, use_skip=False, use_distance=False, use_wln=False,
                 neigh_th=8):
        """
        Construct a MixedNet.
        Channels are split into graphs and diff_block channels, then convoluted, then mixed using GCN
        Parameters:
            C_in_graph (int):                     input dimension
            C_out (int):                    output dimension
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces'].
            (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (float):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0,
            saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient.
            Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(GraphDiffNetBipartite, self).__init__()
        use_gat = True if use_v2 else use_gat

        # # Store parameters

        # Basic parameters
        self.C_in_graph = C_in_graph
        self.C_in_surf = C_in_surf
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block
        self.output_graph = output_graph
        self.neigh_th = neigh_th
        self.use_skip = use_skip

        # Outputs
        self.last_activation = last_activation
        self.dropout = dropout

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
        self.first_lin1 = nn.Linear(C_in_surf, diffnet_width)
        self.first_lin2 = nn.Linear(C_in_graph, diffnet_width)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlockBatch(
                C_width=diffnet_width,
                mlp_hidden_dims=self.mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                use_bn=use_bn
            )
            self.diff_blocks.append(diffnet_block)
            self.add_module("diffnet_block_" + str(i_block), self.diff_blocks[-1])

        self.gcn_blocks = []
        for i_block in range(self.N_block):
            if use_wln:
                gcn_block = WLNConv(node_fdim=diffnet_width, edge_fdim=2, depth=1, hsize=diffnet_width, bias=False,
                                    dropout=dropout, activation='relu')
            else:
                gcn_block = GCN(diffnet_width, diffnet_width, diffnet_width,
                                drate=dropout, use_bn=use_bn, use_distance=use_distance)
            self.gcn_blocks.append(gcn_block)
            self.add_module("gcn_block_" + str(i_block), gcn_block)

        if not use_gat:
            conv_layer = GCNConv
        else:
            conv_layer = GATv2Conv if use_v2 else GATConv
        self.graphsurf_blocks = []
        for i_block in range(self.N_block):
            graphsurf_block = conv_layer(diffnet_width,
                                         diffnet_width if i_block < self.N_block - 1 else C_out,
                                         add_self_loops=True,
                                         edge_dim=1 if use_v2 else None,
                                         )
            self.graphsurf_blocks.append(graphsurf_block)
            self.add_module("graphsurf_block_" + str(i_block), graphsurf_block)

        self.surfgraph_blocks = []
        for i_block in range(self.N_block):
            surfgraph_block = conv_layer(diffnet_width,
                                         diffnet_width if i_block < self.N_block - 1 else C_out,
                                         add_self_loops=True,
                                         edge_dim=1 if use_v2 else None,
                                         )
            self.surfgraph_blocks.append(surfgraph_block)
            self.add_module("surfgraph_block_" + str(i_block), surfgraph_block)

    def forward(self, graph=None, surface=None):
        """
        A forward pass on the MixedNet.
        """
        x_in, mass, L, evals, evecs, gradX, gradY = surface.x, surface.mass, surface.L, surface.evals, surface.evecs, surface.gradX, surface.gradY
        vertices = surface.vertices
        mass = [m.unsqueeze(0) for m in mass]
        L = [ll.unsqueeze(0) for ll in L]
        evals = [e.unsqueeze(0) for e in evals]
        evecs = [e.unsqueeze(0) for e in evecs]

        bipartite_graphsurf, bipartite_surfgraph = compute_bipartite_graphs(vertices, graph, neigh_th=self.neigh_th)

        # Apply the first linear layer
        split_sizes = [tensor.size(0) for tensor in x_in]
        diff_x = torch.split(self.first_lin1(torch.cat(x_in, dim=0)), split_sizes, dim=0)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        NL = len(self.gcn_blocks) - 1
        for layer_i, (graph_block, diff_block,
                      graphsurf_block, surfgraph_block) in enumerate(zip(self.gcn_blocks,
                                                                         self.diff_blocks,
                                                                         self.graphsurf_blocks,
                                                                         self.surfgraph_blocks)):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = graph_block(graph)

            # Now use this for message passing.
            input_feats = [torch.cat((diff, mini_graph.x)) for diff, mini_graph in zip(diff_x, graph.to_data_list())]
            out_surf = [graphsurf_block(input_feat, bigraphsurf.edge_index, bigraphsurf.edge_weight)
                        for input_feat, bigraphsurf in zip(input_feats, bipartite_graphsurf)]

            out_graph = [surfgraph_block(input_feat, bisurfgraph.edge_index, bisurfgraph.edge_weight)
                         for input_feat, bisurfgraph in zip(input_feats, bipartite_surfgraph)]

            diff_x_out = [out[:len(vert)] for out, vert in zip(out_surf, vertices)]
            graph_x_out = torch.cat([out[len(vert):] for out, vert in zip(out_graph, vertices)], dim=0)

            diff_x = [x + y for x, y in zip(diff_x, diff_x_out)] if (self.use_skip and layer_i < NL) else diff_x_out
            graph.x = graph.x + graph_x_out if (self.use_skip and layer_i < NL) else graph_x_out

        if self.output_graph:
            x_out = [mini_graph.x for mini_graph in graph.to_data_list()]
        else:
            x_out = diff_x
        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = [self.last_activation(x) for x in x_out]
        return x_out


class AtomNetGraph(torch.nn.Module):
    def __init__(self, C_in, C_out, C_width, last_factor=2, use_distance=False):
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
        self.use_distance = use_distance

    def forward(self, graph, *largs, **kwargs, ):
        x, edge_index = graph.x, graph.edge_index
        edge_weight = graph.edge_attr if self.use_distance else None
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


def attention_vanilla(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    result = torch.einsum("bhnm,bdhm->bdhn", prob, value)
    return result, prob


def attention_flash(query, key, value):
    query, key, value = query.to(dtype=torch.float16), key.to(dtype=torch.float16), value.to(dtype=torch.float16)
    query, key, value = query.permute(0, 3, 2, 1), key.permute(0, 3, 2, 1), value.permute(0, 3, 2, 1)
    result = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
    result = result.permute(0, 3, 2, 1).float()
    return result, None


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int, flash=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.flash = flash

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [ll(x).view(batch_dim, self.dim, self.num_heads, -1) for ll, x in
                             zip(self.proj, (query, key, value))]
        if self.flash:
            x, _ = attention_flash(query, key, value)
        else:
            x, _ = attention_vanilla(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, flash=True):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim, flash=flash)
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


class GraphDiffNetAttention(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, dropout=0.5,
                 with_gradient_features=True, with_gradient_rotations=True, diffusion_method="spectral", use_bn=True,
                 use_distance=False, output_graph=False, flash=True):
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
            dropout (float):                 if True, internal MLPs use dropout (default: True)
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
        self.output_graph = output_graph

        # Outputs
        self.last_activation = last_activation
        self.dropout = dropout

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
        self.first_lin1 = nn.Linear(5, diffnet_width)
        self.first_lin2 = nn.Linear(C_in, diffnet_width)
        self.last_lin = nn.Linear(diffnet_width, C_out)

        # DiffusionNet blocks
        self.mlp_hidden_dims = [diffnet_width, diffnet_width]
        self.diff_blocks = []
        for i_block in range(self.N_block):
            diffnet_block = DiffusionNetBlockBatch(
                C_width=diffnet_width,
                mlp_hidden_dims=self.mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                use_bn=use_bn
            )

            self.diff_blocks.append(diffnet_block)
            self.add_module("diffnet_block_" + str(i_block), self.diff_blocks[-1])

        self.gcn_blocks = []
        for i_block in range(self.N_block):
            gcn_block = GCN(diffnet_width, diffnet_width, diffnet_width,
                            drate=dropout, use_bn=use_bn, use_distance=use_distance)
            self.gcn_blocks.append(gcn_block)
            self.add_module("gcn_block_" + str(i_block), gcn_block)

        self.att_graph_blocks = []
        for i_block in range(self.N_block):
            att_block = AttentionalPropagation(diffnet_width, num_heads=4, flash=flash)
            self.att_graph_blocks.append(att_block)
            self.add_module("att_graph_blocks_" + str(i_block), att_block)

        self.att_diff_blocks = []
        for i_block in range(self.N_block):
            att_block = AttentionalPropagation(diffnet_width, num_heads=4, flash=flash)
            self.att_diff_blocks.append(att_block)
            self.add_module("att_diff_blocks_" + str(i_block), att_block)

    def forward(self, graph=None, surface=None):
        """
        A forward pass on the MixedNet.
        """

        x_in, mass, L, evals, evecs, gradX, gradY = surface.x, surface.mass, surface.L, surface.evals, surface.evecs, surface.gradX, surface.gradY
        mass = [m.unsqueeze(0) for m in mass]
        L = [ll.unsqueeze(0) for ll in L]
        evals = [e.unsqueeze(0) for e in evals]
        evecs = [e.unsqueeze(0) for e in evecs]

        # Apply the first linear layer
        split_sizes = [tensor.size(0) for tensor in x_in]
        diff_x = torch.split(self.first_lin1(torch.cat(x_in, dim=0)), split_sizes, dim=0)
        graph.x = self.first_lin2(graph.x)

        # Apply each of the blocks
        for graph_block, diff_block, att_graph_block, att_diff_block in zip(self.gcn_blocks, self.diff_blocks,
                                                                            self.att_graph_blocks,
                                                                            self.att_diff_blocks):
            diff_x = diff_block(diff_x, mass, L, evals, evecs, gradX, gradY)
            graph.x = graph_block(graph)
            diff_x = [att_diff_block(diff_x[i], mini_graph.x) for i, mini_graph in enumerate(graph.to_data_list())]
            # todo check if this is correct (or should we use .from_data_list)
            graph.x = torch.cat(
                [att_graph_block(mini_graph.x, diff_x[i]) for i, mini_graph in enumerate(graph.to_data_list())], dim=0)

        if self.output_graph:
            x_out = [mini_graph.x for mini_graph in graph.to_data_list()]
        else:
            x_out = diff_x
        # Apply last linear :
        x_out = [self.last_lin(x) for x in x_out]

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = [self.last_activation(x) for x in x_out]

        return x_out
