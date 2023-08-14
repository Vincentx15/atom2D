from scipy.spatial.transform import Rotation as R
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
        if self.add_xyz and data is not None:
            verts, x = data.vertices.clone(), data.x
            verts = center_normalize([verts])[0]
            data.x = torch.cat((verts, x), dim=1)

        return data


class AddXYZRotationTransform(object):
    def __init__(self, add_xyz=False):
        self.add_xyz = add_xyz

    def __call__(self, data):
        if self.add_xyz and data is not None:
            is_right = hasattr(data, 'locs_right')
            locs = data.locs_right if is_right else data.locs_left
            verts, x = data.vertices.clone(), data.x
            verts, locs = center_normalize([verts], [locs])
            verts, locs = verts[0], locs[0]

            rot_mat = torch.from_numpy(R.random().as_matrix()).float()
            verts = verts @ rot_mat
            locs = locs @ rot_mat if is_right else locs
            data.x = torch.cat((verts, x), dim=1)
            if is_right:
                data.locs_right = locs
            else:
                data.locs_left = locs

        return data


class AddMSPTransform(object):
    def __init__(self, add_xyz=False):
        self.add_xyz = add_xyz

    def __call__(self, data):
        if self.add_xyz and data is not None:
            has_coord = hasattr(data, 'coords')
            verts, x, all_vertices = data.vertices.clone(), data.x, data.all_vertices

            if has_coord:
                coords = data.coords
                _, (verts, coords) = center_normalize(all_vertices, [verts, coords])
                data.coords = coords
            else:
                verts = center_normalize(all_vertices, [verts])[1][0]

            data.x = torch.cat((verts, x), dim=1)

        return data