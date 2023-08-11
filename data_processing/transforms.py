import torch


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


class AddXYZTransform(object):
    def __init__(self, add_xyz=False):
        self.add_xyz = add_xyz

    def __call__(self, data):
        if self.add_xyz:
            verts, x = data.vertices.clone(), data.x
            verts = center_normalize([verts])[0]
            data.x = torch.cat((verts, x), dim=1)

        return data
