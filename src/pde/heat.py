import numpy as np
import torch
from scipy import interpolate

import deepxde as dde
from . import baseclass
from ..utils.func_cache import cache_tensor
from ..utils.geom import CSGMultiDifference
from ..utils.random import generate_heat_2d_coef


class Heat2D_VaryingCoef(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/heat_darcy.dat", bbox=[0, 1, 0, 1, 0, 5], A=200, m=(1, 5, 1)):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=(self.bbox[0], self.bbox[2]), xmax=(self.bbox[1], self.bbox[3]))
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # PDE
        # self.heat_2d_coef = generate_heat_2d_coef(N_res=256, alpha=4, bbox=bbox[0:4])
        self.heat_2d_coef = np.loadtxt("ref/heat_2d_coef_256.dat")

        @cache_tensor
        def coef(x):
            return torch.Tensor(
                interpolate.griddata(self.heat_2d_coef[:, 0:2], self.heat_2d_coef[:, 2], x.detach().cpu().numpy()[:, 0:2], method='nearest')
            ).unsqueeze(dim=1)

        def heat_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
            u_t = dde.grad.jacobian(u, x, i=0, j=2)

            def f(x):
                return A * torch.sin(m[0] * torch.pi * x[:, 0:1]) * torch.sin(m[1] * torch.pi * x[:, 1:2]) * torch.sin(m[2] * torch.pi * x[:, 2:3])

            return [u_t - coef(x) * u_xx - f(x)]

        self.pde = heat_pde
        self.set_pdeloss(num=1)

        self.load_ref_data(datapath)

        # BCs
        def boundary_t0(x, on_initial):
            return on_initial and np.isclose(x[2], bbox[4])

        def boundary_xb(x, on_boundary):
            return on_boundary and (np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]) or np.isclose(x[1], bbox[2]) or np.isclose(x[1], bbox[3]))

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_xb,
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points(mul=4)


class Heat2D_Multiscale(baseclass.BaseTimePDE):

    def __init__(
        self, 
        bbox=[0, 1, 0, 1, 0, 5], 
        pde_coef=(1 / np.square(500 * np.pi), 1 / np.square(np.pi)), 
        init_coef=(20 * np.pi, np.pi),
    ):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_t = dde.grad.jacobian(u, x, j=2)

            return [u_t - pde_coef[0] * u_xx - pde_coef[1] * u_yy]

        self.pde = pde
        self.set_pdeloss(num=1)

        def ref_sol(xt):
            return np.sin(init_coef[0] * xt[:, 0:1]) * np.sin(init_coef[1] * xt[:, 1:2]) * \
                   np.exp(-(pde_coef[0]*init_coef[0]**2 + pde_coef[1]*init_coef[1]**2) * xt[:, 2:3])

        self.ref_sol = ref_sol

        # BCs

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points(mul=4)


class Heat2D_ComplexGeometry(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/heat_complex.dat", bbox=[-8, 8, -12, 12, 0, 3]):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        rec = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        # big circles
        big_circles = []
        big_centers = [(-4, -3), (4, -3), (-4, 3), (4, 3), (-4, -9), (4, -9), (-4, 9), (4, 9), (0, 0), (0, 6), (0, -6)]
        for center in big_centers:
            big_circles.append(dde.geometry.Disk(center, radius=1))
        # small circles
        small_circles = []
        small_centers = [(-3.2, -6), (-3.2, 6), (3.2, -6), (3.2, 6), (-3.2, 0), (3.2, 0)]
        for center in small_centers:
            small_circles.append(dde.geometry.Disk(center, radius=0.4))

        self.geom = CSGMultiDifference(rec, big_circles + small_circles)
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.geomtime = geomtime

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_t = dde.grad.jacobian(u, x, j=2)

            return [u_t - u_xx - u_yy]

        self.pde = pde
        self.set_pdeloss(num=1)

        self.load_ref_data(datapath)

        def is_on_big_circle(x):
            for circle in big_circles:
                if circle.on_boundary(x[0:2]):
                    return True
            return False

        def is_on_small_circle(x):
            for circle in small_circles:
                if circle.on_boundary(x[0:2]):
                    return True
            return False

        # NOTE: the sign of RobinBC needs to be checked
        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _, u: 5 - u),
            'bc': (lambda x, on_boundary: on_boundary and is_on_big_circle(x)),
            'type': 'robin'
        }, {
            'component': 0,
            'function': (lambda _, u: 1 - u),
            'bc': (lambda x, on_boundary: on_boundary and is_on_small_circle(x)),
            'type': 'robin'
        }, {
            'component': 0,
            'function': (lambda _, u: 0.1 - u),
            'bc': (lambda x, on_boundary: on_boundary and not is_on_big_circle(x) and not is_on_small_circle(x)),
            'type': 'robin'
        }])

        # Training Config
        self.training_points(mul=4)


class Heat2D_LongTime(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/heat_longtime.dat", bbox=[0, 1, 0, 1, 0, 100], k=1, m1=4, m2=2):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # pde
        INITIAL_COEF_1 = 4 * np.pi
        INITIAL_COEF_2 = 3 * np.pi

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_t = dde.grad.jacobian(u, x, j=2)

            return [
                u_t - 0.001 * (u_xx + u_yy) - 5 * torch.sin(k * torch.square(u)) *
                (1 + 2 * torch.sin(x[:, 2:3] * np.pi / 4)) * torch.sin(m1 * np.pi * x[:, 0:1]) * torch.sin(m2 * np.pi * x[:, 1:2])
            ]

        self.pde = pde
        self.set_pdeloss(num=1)

        self.load_ref_data(datapath)

        # BCs
        def f_func(x):
            return np.sin(INITIAL_COEF_1 * x[:, 0:1]) * np.sin(INITIAL_COEF_2 * x[:, 1:2])

        self.add_bcs([{
            'component': 0,
            'function': f_func,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points(mul=4)


class HeatND(baseclass.BaseTimePDE):

    def __init__(self, dim=5, T=1):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = [-1, 1] * dim + [0, T]
        self.geom = dde.geometry.Hypersphere([0] * dim, 1)
        timedomain = dde.geometry.TimeDomain(0, T)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # pde
        def xnorm(xt):
            return (xt[:, :-1]**2).sum(axis=1).reshape(-1, 1)

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            for i in range(1, dim):
                u_xx = u_xx + dde.grad.hessian(u, x, i=i, j=i)
            u_t = dde.grad.jacobian(u, x, i=0, j=dim)

            def f(xt):
                x2, t = xnorm(xt), xt[:, -1:]
                return -1 / dim * x2 * torch.exp(x2 / 2 + t)

            return 1 / dim * u_xx + f(x) - u_t

        self.pde = pde
        self.set_pdeloss(num=1)

        def ref_sol(xt):
            x2, t = xnorm(xt), xt[:, -1:]
            return np.exp(x2 / 2 + t)

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'neumann',
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic',
        }])

        self.training_points(mul=4)
