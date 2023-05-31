import torch


def unwrap_feats(geom_feat, device='cpu'):
    features, confidence, vertices, mass, L, evals, evecs, gradX, gradY, faces = geom_feat
    features = torch.cat((features, confidence[..., None]), dim=-1)
    gradX, gradY = gradX.to_sparse(), gradY.to_sparse()
    dict_return = {'x_in': features,
                   'mass': mass,
                   'L': L,
                   'evals': evals,
                   'evecs': evecs,
                   'gradX': gradX,
                   'gradY': gradY,
                   'vertices': vertices,
                   'faces': faces}
    dict_return_32 = {k: v.float().to(device) for k, v in dict_return.items()}
    return dict_return_32


def center_normalize(verts1, verts2):
    """
    Center and normalize the vertices for the whole system
    """
    verts = torch.cat([verts1, verts2], dim=0)
    mean = verts.mean(dim=0)
    verts = verts - mean
    max_dist = verts.norm(dim=1).max()
    verts = verts / max_dist
    verts1, verts2 = verts[:verts1.shape[0]], verts[verts1.shape[0]:]
    return verts1, verts2


def list_to_numpy(tensor_list):
    return [to_numpy(x) for x in tensor_list]


def list_from_numpy(tensor_list, device='cpu'):
    return [from_numpy(x, device=device) for x in tensor_list]


def to_numpy(x):
    return x.detach().cpu().numpy()


def from_numpy(x, device=None):
    return torch.from_numpy(x).to(device)
