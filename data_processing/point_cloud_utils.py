import Bio.PDB as bio
import matplotlib.pyplot as plt
import numpy as np
import torch

ELEMENT_MAPPING = {'C': 0, 'O': 1, 'N': 2, 'S': 3}


def torch_rbf(points_1, points_2, feats_1, sigma=2.5, eps=0.01):
    """
    Get the signal on the points1 onto the points 2
    :param points_1: n,3
    :param points_2:m,3
    :param feats_1:n,k
    :param sigma:
    :return: m,k
    """
    if not points_1.dtype == points_2.dtype == feats_1.dtype:
        raise ValueError(f"can't RBF with different dtypes{points_1.dtype, points_2.dtype, feats_1.dtype}")
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
        rbf_weights = torch.div(rbf_weights, line_norms[:, None])
    feats_2 = torch.mm(rbf_weights, feats_1)
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


def get_features_pdb(pdb):
    """
    Transform pdb data into a Nx3 coords and Kx3 feature vector
    :param pdb:
    :return:
    """
    parser = bio.MMCIFParser() if pdb.endswith('.cif') else bio.PDBParser()
    structure = parser.get_structure("", pdb)
    all_coords = []
    all_features = []
    # Now loop through atoms and get the encodings
    for i, atom in enumerate(structure.get_atoms()):
        residue = atom.get_parent()
        if residue.id[0] == " ":
            elem = atom.element
            if elem in ELEMENT_MAPPING:
                coord_x = atom.get_coord()
                feature_x = ELEMENT_MAPPING[elem]
                all_coords.append(coord_x)
                all_features.append(feature_x)
    all_coords = np.asarray(all_coords)
    all_features = np.asarray(all_features, dtype=np.int8)

    # Finally one hot encode it
    encoded_all_features = np.zeros((all_features.size, len(ELEMENT_MAPPING)), dtype=np.float32)
    encoded_all_features[np.arange(all_features.size), all_features] = 1
    return all_coords, encoded_all_features


def get_features(pdb, verts, sigma=3.):
    """
    Project the PDB info onto a set of vertices
    :param pdb:
    :param verts:
    :param sigma:
    :return:
    """
    coords_pdb, feats_pdb = get_features_pdb(pdb)
    verts = np.asarray(verts, np.float32)
    verts_feats, confidence = numpy_rbf(feats_1=feats_pdb, points_1=coords_pdb, points_2=verts, sigma=sigma)
    return verts_feats, confidence


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
    points_1 = torch.randn(size=(2000, 1)) * 10
    points_2 = torch.randn(size=(50, 1)) * 20
    feats_1 = torch.sin(points_1)
    feats_2, confidence = torch_rbf(points_1=points_1, points_2=points_2, feats_1=feats_1, sigma=0.5)

    plt.scatter(points_1.numpy(), feats_1.numpy())
    plt.scatter(points_2.numpy(), feats_2.numpy())
    plt.scatter(points_2.numpy(), torch.tanh(confidence).numpy())
    plt.show()

    # Actual PDB example
    # verts, faces = surface_utils.read_face_and_triangles('../data/example_files/4kt3_mesh.ply')
    # feats, confidence = get_features(pdb='../data/example_files/4kt3.pdb', verts=verts, sigma=3)
    # plt.hist(confidence)
    # plt.show()
