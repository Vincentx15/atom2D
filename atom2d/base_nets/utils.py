import torch
from torch_geometric.data import Data


def create_pyg_graph_object(coords, features, sigma=3):
    """
    Define a simple fully connected graph from a set of coords
    :param coords:
    :param features:
    :param sigma:
    :return:
    """
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
