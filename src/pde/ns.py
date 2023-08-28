import numpy as np
import torch

import deepxde as dde
from . import baseclass


class NS2D_Classic(baseclass.BasePDE):

    def __init__(self, datapath="ref/ns2d.dat", nu=1, bbox=[0, 8, 0, 8], circles=[[6, 6, 1.0], [2, 1.0, 0.5]], linear=False):
        # bbox: left(x[0]) right(x[0]) bottom(x[1]) top(x[1])
        # circles[]: center0 center1 radius
        super().__init__()
        self.nu = nu
        self.bbox = bbox
        self.circles = circles
        # output dim
        self.output_config = [{'name': s} for s in ['u', 'v', 'p']]
        # geom
        rec = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        geom = rec
        disks = []
        for i in range(len(circles)):
            disk = dde.geometry.Disk(circles[i][0:2], circles[i][2])
            disks.append(disk)
            geom = dde.geometry.csg.CSGDifference(geom, disk)
        self.geom = geom

        # PDE
        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - self.nu * (u_vel_xx + u_vel_yy))
            momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - self.nu * (v_vel_xx + v_vel_yy))
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        def ns_pde_linear(x, u):
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = p_x - nu * (u_vel_xx + u_vel_yy)
            momentum_y = p_y - nu * (v_vel_xx + v_vel_yy)
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde if not linear else ns_pde_linear
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])

        self.load_ref_data(datapath)

        def boundary_left(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[0])

        def boundary_right(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[1])

        def boundary_up_down(x, on_boundary):
            return on_boundary and (np.isclose(x[1], bbox[2]) or np.isclose(x[1], bbox[3]))

        def boundary_circle(x, on_boundary):
            return on_boundary and (not rec.on_boundary(x))

        def u_func(x):
            return x[:, 1:2] * (bbox[3] - x[:, 1:2]) / (2 * bbox[3])

        self.add_bcs([{
            'component': 0,
            'function': u_func,
            'bc': boundary_left,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_left,
            'type': 'dirichlet'
        }, {
            'component': 2,
            'function': (lambda _: 0),
            'bc': boundary_right,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_up_down,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_up_down,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_circle,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_circle,
            'type': 'dirichlet'
        }])

        # traing config
        self.training_points()


class NS2D_LidDriven(baseclass.BasePDE):

    def __init__(self, datapath="ref/lid_driven_a4.dat", a=4, nu=1 / 100, bbox=[0, 1, 0, 1]):
        super().__init__()
        # output dim
        self.output_config = [{'name': s} for s in ['u', 'v', 'p']]
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])

        # PDE
        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
            momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy))
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])

        self.load_ref_data(datapath)

        # bc
        def boundary_top(x, on_boundary):
            return on_boundary and np.isclose(x[1], bbox[3])

        def boundary_not_top(x, on_boundary):
            return on_boundary and not np.isclose(x[1], bbox[3])

        self.add_bcs([{
            'component': 0,
            'function': (lambda x: a * x[:, 0:1] * (1 - x[:, 0:1])),
            'bc': boundary_top,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_top,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_not_top,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_not_top,
            'type': 'dirichlet'
        }, {
            'component': 2,
            "points": np.array([[0, 0]]),
            'values': np.array([[0]]),
            'type': 'pointset'
        }])

        # training config
        self.training_points()


class NS2D_BackStep(baseclass.BasePDE):

    def __init__(
        self,
        datapath="ref/ns_0_obstacle.dat",
        nu=1 / 100,
        bbox=[0, 4, 0, 2],
        # obstacle={
        #     'circ': [(1, 0.5, 0.2), (2, 0.5, 0.3), (2.7, 1.6, 0.2)],
        #     'rec': [(2.8, 3.6, 0.8, 1.1)]
        # },
        obstacle={},
    ):
        super().__init__()
        # output dim
        self.output_config = [{'name': s} for s in ['u', 'v', 'p']]
        # geom
        eps = 1e-5
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        rec = dde.geometry.Rectangle(xmin=[bbox[0] - eps, bbox[3] / 2], xmax=[bbox[1] / 2, bbox[3] + eps])
        self.geom = dde.geometry.csg.CSGDifference(self.geom, rec)

        # TODO: use CSGMultiDifference
        for name, configs in obstacle.items():
            if name == 'circ':
                for c in configs:
                    diff_geom = dde.geometry.Disk(c[:2], c[2])
                    self.geom = dde.geometry.csg.CSGDifference(self.geom, diff_geom)
            elif name == 'rec':
                for c in configs:
                    diff_geom = dde.geometry.Rectangle(xmin=[c[0], c[2]], xmax=[c[1], c[3]])
                    self.geom = dde.geometry.csg.CSGDifference(self.geom, diff_geom)

        # PDE
        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
            momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy))
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])

        self.load_ref_data(datapath)

        # bcs
        def boundary_in(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[0])

        def boundary_out(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[1])

        def boundary_other(x, on_boundary):
            return on_boundary and not (boundary_in(x, on_boundary) or boundary_out(x, on_boundary))

        def u_func(x):
            return x[:, 1:2] * (1 - x[:, 1:2]) * 4

        self.add_bcs([{
            'component': 0,
            'function': u_func,
            'bc': boundary_in,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_in,
            'type': 'dirichlet'
        }, {
            'component': 2,
            'function': (lambda _: 0),
            'bc': boundary_out,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_other,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_other,
            'type': 'dirichlet'
        }])

        # traing config
        self.training_points()


class NS2D_LongTime(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/ns_long.dat", nu=1 / 100, bbox=[0, 2, 0, 1, 0, 5]):
        super().__init__()
        # output dim
        self.output_config = [{'name': s} for s in ['u', 'v', 'p']]
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # PDE
        COEF_A1 = 1
        COEF_A2 = 1
        COEF_A3 = 1

        def initial_fy(x):
            return -torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2]) * torch.sin(np.pi * x[:, 2:3])

        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            u_t = dde.grad.jacobian(u, x, i=0, j=2)
            v_t = dde.grad.jacobian(u, x, i=1, j=2)

            momentum_x = (u_t + u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
            momentum_y = (v_t + u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy) - initial_fy(x))
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])

        self.load_ref_data(datapath)

        # BC
        def boundary_in(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[0])

        def boundary_out(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[1])

        def boundary_other(x, on_boundary):
            return on_boundary and not (boundary_in(x, on_boundary) or boundary_out(x, on_boundary))

        def u_func(x):
            y, t = x[:, 1:2], x[:, 2:3]
            return np.sin(np.pi * y) * (COEF_A1 * np.sin(np.pi * t) + COEF_A2 * np.sin(3 * np.pi * t) + COEF_A3 * np.sin(5 * np.pi * t))

        self.add_bcs([{
            'component': 0,
            'function': u_func,
            'bc': boundary_in,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_in,
            'type': 'dirichlet'
        }, {
            'component': 2,
            'function': (lambda _: 0),
            'bc': boundary_out,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_other,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_other,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 2,
            'function': (lambda _: 0),
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }])

        # training config
        self.training_points(mul=4)
