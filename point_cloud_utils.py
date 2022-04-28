import Bio.PDB as bio
import matplotlib.pyplot as plt
import numpy as np
import torch

element_mapping = {'C': 0, 'O': 1, 'N': 2, 'S': 3}


def torch_rbf(points_1, points_2, feats_1, sigma=2.5, eps=0.01):
    """
    Get the signal on the points1 onto the points 2
    :param points_1: n,3
    :param points_2:m,3
    :param feats_1:n,k
    :param sigma:
    :return: m,k
    """
    # Get all to all dists, make it a weight and message passing.
    #     TODO : Maybe include sigma as a learnable parameter
    with torch.no_grad():
        all_dists = torch.cdist(points_2, points_1)
        rbf_weights = torch.exp(-all_dists / sigma)
        # TODO : Probably some speedup is possible with sparse
        # rbf_weights_selection = rbf_weights > (np.exp(-2 * sigma))
        # rbf_weights = rbf_weights_selection * rbf_weights

        # Then normalize by line, the sum are a confidence score
        line_norms = torch.sum(rbf_weights, dim=1) + eps
        feats_2 = torch.mm(rbf_weights, feats_1)
        feats_2 = torch.div(feats_2, line_norms[:, None])
    return feats_2, line_norms


def numpy_rbf(points_1, points_2, feats_1, sigma=2.5, eps=0.01):
    """
    Just a numpy wrapper, the underlying code stays in torch
    """
    points_1 = torch.from_numpy(points_1)
    points_2 = torch.from_numpy(points_2)
    feats_1 = torch.from_numpy(feats_1)
    feats_2, line_norms = torch_rbf(points_1=points_1, points_2=points_2, feats_1=feats_1, sigma=sigma, eps=eps)
    return feats_2.numpy(), line_norms.numpy()


def get_features(pdb, verts):
    parser = bio.MMCIFParser() if pdb.endswith('.cif') else bio.PDBParser()
    structure = parser.get_structure("", pdb)
    all_coords=[]
    all_features=[]
    for i, atom in enumerate(structure.get_atoms()):
        residue = atom.get_parent()
        if residue.id[0] == " ":
            print(atom.get_name())
            coord = atom.get_coord()


if __name__ == '__main__':
    pass
    # # decoys in higher dims
    # points_1 = torch.randn(size=(10, 3)) * 10
    # points_2 = torch.randn(size=(8, 3)) * 10
    # feats_1 = torch.randn(size=(10, 7))
    # # # Now we want to send the content in point_1, feats_2 to points_2.
    # feats_2, confidence = torch_rbf(points_1=points_1, points_2=points_2, feats_1=feats_1)
    # print(feats_2.shape, confidence.shape)

    # # Visual proof
    # points_1 = torch.randn(size=(2000, 1)) * 10
    # points_2 = torch.randn(size=(30, 1)) * 15
    # feats_1 = torch.sin(points_1)
    # feats_2, confidence = torch_rbf(points_1=points_1, points_2=points_2, feats_1=feats_1, sigma=0.5)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.scatter(points_1.numpy(), feats_1.numpy())
    # plt.scatter(points_2.numpy(), feats_2.numpy())
    # plt.scatter(points_2.numpy(), torch.tanh(confidence).numpy())
    # plt.show()

    feats = get_features(pdb='data/example_files/4kt3.pdb', verts=None)
