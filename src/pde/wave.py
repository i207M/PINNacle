import numpy as np
import torch
from scipy import interpolate

import deepxde as dde
from . import baseclass
from ..utils.random import generate_darcy_2d_coef
from ..utils.func_cache import cache_tensor


class Wave1D(baseclass.BasePDE):

    def __init__(self, C=2, bbox=[0, 1, 0, 1], scale=1, a=4):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = [0, scale, 0, scale]
        self.geom = dde.geometry.Rectangle(xmin=[self.bbox[0], self.bbox[2]], xmax=[self.bbox[1], self.bbox[3]])

        # define PDE
        def wave_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_tt = dde.grad.hessian(u, x, i=1, j=1)

            return u_tt - C**2 * u_xx

        self.pde = wave_pde
        self.set_pdeloss(num=1)

        def ref_sol(x):
            x = x / scale
            return (np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(a * np.pi * x[:, 0:1]) * np.cos(2 * a * np.pi * x[:, 1:2]))

        self.ref_sol = ref_sol

        def boundary_x0(x, on_boundary):
            return on_boundary and (np.isclose(x[0], self.bbox[0]) or np.isclose(x[0], self.bbox[1]))

        def boundary_t0(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.bbox[2])

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'neumann'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': boundary_t0,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': boundary_x0,
            'type': 'dirichlet'
        }])

        # training config
        self.training_points()


class Wave2D_Heterogeneous(baseclass.BasePDE):

    def __init__(self, datapath="ref/wave_darcy.dat", bbox=[-1, 1, -1, 1, 0, 5], mu=(-0.5, 0), sigma=0.3):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        # NOTE: no circ are deleted, since the pde is currently not regraded as TimePDE and 3D-CSGDifference is difficult)
        self.bbox = bbox
        self.geom = dde.geometry.Hypercube(xmin=(self.bbox[0], self.bbox[2], self.bbox[4]), xmax=(self.bbox[1], self.bbox[3], self.bbox[5]))

        # PDE
        # self.darcy_2d_coef = generate_darcy_2d_coef(N_res=256, alpha=4, bbox=bbox[0:4])
        self.darcy_2d_coef = np.loadtxt("ref/darcy_2d_coef_256.dat")

        @cache_tensor
        def coef(x):
            return torch.Tensor(
                interpolate.griddata(self.darcy_2d_coef[:, 0:2], self.darcy_2d_coef[:, 2], (x.detach().cpu().numpy()[:, 0:2] + 1) / 2)
            ).unsqueeze(dim=-1)

        def wave_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
            u_tt = dde.grad.hessian(u, x, i=2, j=2)

            return u_xx - u_tt / coef(x)

        self.pde = wave_pde
        self.set_pdeloss(num=1)

        self.load_ref_data(datapath, t_transpose=True)

        # BCs
        def boundary_t0(x, on_initial):
            return np.isclose(x[2], bbox[4])

        def boundary_rec(x, on_boundary):
            return on_boundary and (np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]) or np.isclose(x[1], bbox[2]) or np.isclose(x[1], bbox[3]))

        def initial_condition(x):
            return np.exp(-((x[:, 0:1] - mu[0])**2 + (x[:, 1:2] - mu[1])**2) / (2 * sigma**2))

        self.add_bcs([{
            'component': 0,
            'function': initial_condition,
            'bc': boundary_t0,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'neumann'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_rec,
            'type': 'neumann',
        }])

        # training config
        self.training_points(mul=4)


class Wave2D_LongTime(baseclass.BaseTimePDE):

    def __init__(self, bbox=[0, 1, 0, 1, 0, 100], a=np.sqrt(2), m1=1, m2=3, n1=1, n2=2, p1=1, p2=1):
        super().__init__()

        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # pde
        INITIAL_COEF_1 = 1
        INITIAL_COEF_2 = 1

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_tt = dde.grad.hessian(u, x, i=2, j=2)

            return [u_tt - (u_xx + a * a * u_yy)]

        self.pde = pde
        self.set_pdeloss(num=1)

        # BCs
        def ref_sol(x):
            return (
                INITIAL_COEF_1 * np.sin(m1 * np.pi * x[:, 0:1]) * np.sinh(n1 * np.pi * x[:, 1:2]) * np.cos(p1 * np.pi * x[:, 2:3])
                + INITIAL_COEF_2 * np.sinh(m2 * np.pi * x[:, 0:1]) * np.sin(n2 * np.pi * x[:, 1:2]) * np.cos(p2 * np.pi * x[:, 2:3])
            )

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        # training config
        self.training_points(mul=4)
