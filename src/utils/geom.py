import numpy as np

from deepxde import config
from deepxde.geometry import geometry


class CSGMultiDifference(geometry.Geometry):
    """Construct an object by CSG Difference."""

    def __init__(self, geom1, geom2_list):
        super().__init__(geom1.dim, geom1.bbox, geom1.diam)
        self.geom1 = geom1
        self.geom2_list = geom2_list

    def not_inside_geom2(self, x):
        return ~(np.stack([geom2.inside(x) for geom2 in self.geom2_list], axis=1).any(axis=1))

    def inside(self, x):
        not_in_geom2 = self.not_inside_geom2(x)
        return np.logical_and(self.geom1.inside(x), not_in_geom2)

    def on_boundary(self, x):
        not_in_geom2 = self.not_inside_geom2(x)
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), not_in_geom2),
            np.logical_and(
                self.geom1.inside(x),
                np.stack([geom2.on_boundary(x) for geom2 in self.geom2_list], axis=1).any(axis=1),
            )
        )

    def random_points(self, n, random='pseudo'):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            tmp = self.geom1.random_points(n, random=random)
            tmp = tmp[self.not_inside_geom2(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def random_boundary_points(self, n, random='pseudo'):
        x = np.empty(shape=(n, self.dim), dtype=config.real(np))
        i = 0
        while i < n:
            geom1_boundary_points = self.geom1.random_boundary_points(n, random=random)
            geom1_boundary_points = geom1_boundary_points[self.not_inside_geom2(geom1_boundary_points)]

            geom2_boundary_points = np.concatenate([geom2.random_boundary_points(n, random=random) for geom2 in self.geom2_list], axis=0)
            geom2_boundary_points = geom2_boundary_points[self.geom1.inside(geom2_boundary_points)]

            tmp = np.concatenate((geom1_boundary_points, geom2_boundary_points))
            tmp = np.random.permutation(tmp)

            if len(tmp) > n - i:
                tmp = tmp[:n - i]
            x[i:i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def boundary_normal(self, x):
        not_in_geom2 = self.not_inside_geom2(x)
        res = np.logical_and(self.geom1.on_boundary(x), not_in_geom2)[:, np.newaxis] * self.geom1.boundary_normal(x)
        for geom2 in self.geom2_list:
            res += np.logical_and(self.geom1.inside(x), geom2.on_boundary(x))[:, np.newaxis] * -geom2.boundary_normal(x)
        return res
