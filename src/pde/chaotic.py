import numpy as np
import deepxde as dde

from . import baseclass


class GrayScottEquation(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/grayscott.dat", bbox=[-1, 1, -1, 1, 0, 200], b=0.04, d=0.1, epsilon=(1e-5, 5e-6)):
        super().__init__()
        # output dim
        self.output_dim = 2

        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle((self.bbox[0], self.bbox[2]), (self.bbox[1], self.bbox[3]))
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # PDE
        def pde(x, y):
            u, v = y[:, 0:1], y[:, 1:2]

            u_t = dde.grad.jacobian(u, x, i=0, j=2)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            v_t = dde.grad.jacobian(v, x, i=0, j=2)
            v_xx = dde.grad.hessian(v, x, i=0, j=0)
            v_yy = dde.grad.hessian(v, x, i=1, j=1)
            return [u_t - (epsilon[0] * (u_xx + u_yy) + b * (1 - u) - u * (v**2)), v_t - (epsilon[1] * (v_xx + v_yy) - d * v + u * (v**2))]

        self.pde = pde
        self.set_pdeloss(num=2)

        self.load_ref_data(datapath, t_transpose=False)

        # BC
        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], bbox[4])

        def ic_func(x, component):
            if component == 0:
                return 1 - np.exp(-80 * ((x[:, 0] + 0.05)**2 + (x[:, 1] + 0.02)**2))
            else:
                return np.exp(-80 * ((x[:, 0] - 0.05)**2 + (x[:, 1] - 0.02)**2))

        self.add_bcs([{
            'component': 0,
            'function': (lambda x: ic_func(x, component=0)),
            'bc': boundary_ic,
            'type': 'ic'
        }, {
            'component': 1,
            'function': (lambda x: ic_func(x, component=1)),
            'bc': boundary_ic,
            'type': 'ic'
        }])

        self.training_points(mul=4)


class KuramotoSivashinskyEquation(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/Kuramoto_Sivashinsky.dat", bbox=[0, 2 * np.pi, 0, 1], alpha=100 / 16, beta=100 / (16 * 16), gamma=100 / (16**4)):
        super().__init__()
        # output dim
        self.output_dim = 1

        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Interval(bbox[0], bbox[1])
        timedomain = dde.geometry.TimeDomain(bbox[2], bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # PDE
        def pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_xxxx = dde.grad.hessian(u_xx, x, i=0, j=0)

            return u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx

        self.pde = pde
        self.set_pdeloss(num=1)

        self.load_ref_data(datapath, t_transpose=False)

        # BCs
        self.add_bcs([{
            'component': 0,
            'function': (lambda x: np.cos(x[:, 0:1]) * (1 + np.sin(x[:, 0:1]))),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }])

        # training point
        self.training_points()
