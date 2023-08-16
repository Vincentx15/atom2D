from scipy.spatial.transform import Rotation as R
import torch


class Normalizer:
    def __init__(self, add_xyz=False):
        """
        Null operation if add xyz is False
        Otherwise, it first needs to be set on some data, then it can be applied to any new data.
        :param add_xyz:
        """
        self.add_xyz = add_xyz
        self.mean = None
        self.rot_mat = torch.from_numpy(R.random().as_matrix()).float()
        self.scale = 30

    def set_mean(self, data):
        """

        :param data: N,3 tensor
        :return:
        """
        if self.add_xyz:
            self.mean = torch.mean(data, dim=0)
        return self

    def transform(self, data):
        if self.add_xyz and data is not None:
            data = data - self.mean / self.scale
            data = data @ self.rot_mat
        return data

    def transform_surface(self, surface):
        if self.add_xyz:
            surface.vertices = self.transform(surface.vertices)
            surface.x = torch.cat((surface.x, surface.vertices), dim=1)
        return surface

    def transform_graph(self, graph):
        if self.add_xyz:
            graph.vertices = self.transform(graph.vertices)
            graph.x = torch.cat((graph.x, graph.vertices), dim=1)
        return graph
