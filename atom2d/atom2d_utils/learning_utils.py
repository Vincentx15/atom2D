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


def list_to_numpy(tensor_list):
    return [to_numpy(x) for x in tensor_list]


def list_from_numpy(tensor_list, device='cpu'):
    return [from_numpy(x, device=device) for x in tensor_list]


def to_numpy(x):
    return x.detach().cpu().numpy()


def from_numpy(x, device=None):
    return torch.from_numpy(x).to(device)
