import numpy as np
import torch

import deepxde as dde
from . import baseclass
from ..model.fnn import PFNN


class PoissonInv(baseclass.BasePDE):
    def __init__(self):
        super().__init__()
        # output dim
        self.output_config = [{"name": s} for s in ["u", "a"]]
        # geom
        self.bbox = [0, 1] * 2
        self.geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

        # PDE
        def pde(x, ua):
            u, a = ua[:, 0:1], ua[:, 1:2]
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_y = dde.grad.jacobian(u, x, i=0, j=1)
            d_au = dde.grad.jacobian(a * u_x, x, i=0, j=0) + dde.grad.jacobian(a * u_y, x, i=0, j=1)
            return d_au + self.f_src(x)

        self.pde = pde
        self.set_pdeloss(num=1)

        self.ref_sol = lambda xy: np.concatenate((self.u_ref(xy), self.a_ref(xy)), axis=1)

        self.recommend_net = dde.nn.PFNN(
            [self.input_dim] + 5 * [[50] * self.output_dim] + [self.output_dim], "tanh", "Glorot normal"
        )

        bc_x = np.linspace(0, 1, 50)
        bc_y = np.linspace(0, 1, 50)
        bc_x, bc_y = np.meshgrid(bc_x, bc_y)
        bc_xy = np.stack((bc_x.reshape(-1), bc_y.reshape(-1))).T
        # fmt: off
        self.add_bcs([{
            'component': 0,
            'points': bc_xy,
            'values': self.u_ref(bc_xy) + np.random.normal(loc=0, scale=0.1, size=(2500, 1)),
            'type': 'pointset',
            'name': 'data_loss',
        }, {
            'component': 1,
            'function': self.a_ref,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet',
            'name': 'bc_a',
        }])
        # fmt: on

        self.training_points()

    @staticmethod
    def a_ref(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return 1 / (1 + x**2 + y**2 + (x - 1) ** 2 + (y - 1) ** 2)

    @staticmethod
    def u_ref(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    @staticmethod
    def f_src(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        # fmt: off
        return (
            2 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y) * PoissonInv.a_ref(xy)
            + 2 * np.pi * (
                (2 * x + 1) * torch.cos(np.pi * x) * torch.sin(np.pi * y)
                + (2 * y + 1) * torch.sin(np.pi * x) * torch.cos(np.pi * y)
            ) * PoissonInv.a_ref(xy) ** 2
        )
        # fmt: on

    @staticmethod
    def a_ref_x(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return (2 - 4 * x) / (2 * x**2 - 2 * x + 2 * y**2 - 2 * y + 3) ** 2

    @staticmethod
    def a_ref_y(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return (2 - 4 * y) / (2 * x**2 - 2 * x + 2 * y**2 - 2 * y + 3) ** 2


class HeatInv(baseclass.BaseTimePDE):
    def __init__(self):
        super().__init__()
        # output dim
        self.output_config = [{"name": s} for s in ["u", "a"]]
        # geom
        self.bbox = [-1, 1] * 2 + [0, 1]
        self.geom = dde.geometry.Rectangle(xmin=[-1, -1], xmax=[1, 1])
        timedomain = dde.geometry.TimeDomain(0, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        def f(xyt):
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            s, c, p = torch.sin, torch.cos, np.pi
            # fmt: off
            return torch.exp(-t) * (
                (4 * p**2 - 1) * s(p * x) * s(p * y)
                + p**2 * (
                    2 * s(p * x) ** 2 * s(p * y) ** 2
                    - c(p * x) ** 2 * s(p * y) ** 2
                    - s(p * x) ** 2 * c(p * y) ** 2
                )
            )
            # fmt: on

        def u_ref(xyt):
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

        def a_ref(xyt):  # irrelevent to time domain
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            return 2 + np.sin(np.pi * x) * np.sin(np.pi * y)

        self.f_src = f
        self.u_ref = u_ref
        self.a_ref = a_ref

        # PDE
        def pde(x, ua):
            u, a = ua[:, 0:1], ua[:, 1:2]
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_t = dde.grad.jacobian(u, x, i=0, j=2)
            d_au = dde.grad.jacobian(a * u_x, x, i=0, j=0) + dde.grad.jacobian(a * u_y, x, i=0, j=1)

            return u_t - d_au - f(x)

        self.pde = pde
        self.set_pdeloss(num=1)

        self.ref_sol = lambda xyt: np.concatenate((u_ref(xyt), a_ref(xyt)), axis=1)

        self.recommend_net = PFNN(
            [self.input_dim] + 5 * [[50] * self.output_dim] + [self.output_dim],
            "tanh",
            "Glorot normal",
            split_mask=np.array([[[1, 1, 1]], [[1, 1, 0]]]),
        )

        data_pts = np.loadtxt("ref/heatinv_points.dat")
        # fmt: off
        self.add_bcs([{
            'component': 0,
            'points': data_pts,
            'values': u_ref(data_pts) + np.random.normal(loc=0, scale=0.1, size=(2500, 1)),
            'type': 'pointset',
            'name': 'data_loss',
        }, {
            'component': 1,
            'function': a_ref,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet',
            'name': 'bc_a',
        }])
        # fmt: on

        self.training_points(mul=4)
