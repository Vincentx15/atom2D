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


def center_normalize(list_verts1, list_verts2=None, scale=30):
    """
    Center and normalize the vertices for the whole system
    list_verts2 should use the mean and scale of list_verts1
    """

    # Get the mean and scale
    verts1 = torch.cat(list_verts1, dim=0)
    mean = torch.mean(verts1, dim=0)
    if not scale:
        scale = (verts1 - mean).norm(dim=1).max()
    # Normalize
    list_verts1 = [(verts - mean) / scale for verts in list_verts1]
    if list_verts2 is None:
        return list_verts1
    else:
        list_verts2 = [(verts - mean) / scale for verts in list_verts2]
        return list_verts1, list_verts2


def list_to_numpy(tensor_list):
    return [to_numpy(x) for x in tensor_list]


def list_from_numpy(tensor_list, device='cpu'):
    return [from_numpy(x, device=device) for x in tensor_list]


def to_numpy(x):
    return x.detach().cpu().numpy()


def from_numpy(x, device=None):
    return torch.from_numpy(x).to(device)
