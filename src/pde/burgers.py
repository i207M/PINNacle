import numpy as np
import deepxde as dde
import scipy

from . import baseclass


class Burgers1D(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/burgers1d.dat", geom=[-1, 1], time=[0, 1], nu=0.01 / np.pi):
        super().__init__()
        # output dim
        self.output_dim = 1
        # domain
        self.geom = dde.geometry.Interval(*geom)
        timedomain = dde.geometry.TimeDomain(*time)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = geom + time

        # PDE
        def burger_pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t + u * u_x - nu * u_xx

        self.pde = burger_pde
        self.set_pdeloss()

        # refdata
        self.load_ref_data(datapath)

        # BCs
        def ic_func(x):
            return np.sin(-np.pi * x[:, 0:1])

        self.add_bcs([{
            'component': 0,
            'function': ic_func,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        # train settings
        self.training_points()  # default


class Burgers2D(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/burgers2d_0.dat", icpath=("ref/burgers2d_init_u_0.dat", "ref/burgers2d_init_v_0.dat"), nu=0.001, L=4, T=1):
        super().__init__()
        # output dim
        self.output_dim = 2
        # domain
        self.bbox = [0, L, 0, L, 0, T]
        self.geom = dde.geometry.Rectangle(self.bbox[0:4:2], self.bbox[1:4:2])
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # PDE
        def burger_pde_2d(x, u):
            u1, u2 = u[:, 0:1], u[:, 1:2]

            u1_x = dde.grad.jacobian(u, x, i=0, j=0)
            u1_y = dde.grad.jacobian(u, x, i=0, j=1)
            u1_t = dde.grad.jacobian(u, x, i=0, j=2)
            u1_xx = dde.grad.hessian(u, x, i=0, j=0, component=0)
            u1_yy = dde.grad.hessian(u, x, i=1, j=1, component=0)

            u2_x = dde.grad.jacobian(u, x, i=1, j=0)
            u2_y = dde.grad.jacobian(u, x, i=1, j=1)
            u2_t = dde.grad.jacobian(u, x, i=1, j=2)
            u2_xx = dde.grad.hessian(u, x, i=0, j=0, component=1)
            u2_yy = dde.grad.hessian(u, x, i=1, j=1, component=1)
            return [u1_t + u1 * u1_x + u2 * u1_y - nu * (u1_xx + u1_yy), u2_t + u1 * u2_x + u2 * u2_y - nu * (u2_xx + u2_yy)]

        self.pde = burger_pde_2d
        self.set_pdeloss(num=2)

        self.load_ref_data(datapath) 

        # BCs
        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], 0)

        def boundary_xb(x, on_boundary):
            return on_boundary and (np.isclose(x[0], 0) + np.isclose(x[0], L))

        def boundary_yb(x, on_boundary):
            return on_boundary and (np.isclose(x[1], 0) + np.isclose(x[1], L))

        self.ics = (np.loadtxt(icpath[0]), np.loadtxt(icpath[1]))

        def ic_func(x, component):
            return scipy.interpolate.LinearNDInterpolator(self.ics[component][:, :2], self.ics[component][:, 2:])(x[:, :2])

        self.add_bcs([
            {
                'component': 0,
                'function': (lambda x: ic_func(x, component=0)),
                'bc': boundary_ic,
                'type': 'ic'
            },
            {
                'component': 1,
                'function': (lambda x: ic_func(x, component=1)),
                'bc': boundary_ic,
                'type': 'ic'
            },
            {
                'component': 0,
                'type': 'periodic',
                'component_x': 0,
                'bc': boundary_xb,
            },
            {
                'component': 1,
                'type': 'periodic',
                'component_x': 0,
                'bc': boundary_xb,
            },
            {
                'component': 0,
                'type': 'periodic',
                'component_x': 1,
                'bc': boundary_yb,
            },
            {
                'component': 1,
                'type': 'periodic',
                'component_x': 1,
                'bc': boundary_yb,
            },
        ])

        # train settings
        self.training_points(mul=4)
