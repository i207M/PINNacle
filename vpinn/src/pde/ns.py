import torch
import vpinn
from .problem import Problem

class NSEquation_LidDriven(Problem):

    def __init__(self, nu=1 / 100, bbox=[0, 1, 0, 1], a=2, device='cpu'):
        super().__init__(device)
        # output dim
        self.indim = 2
        self.outdim = 3
        # geom
        self.bbox = bbox
        self.geom = vpinn.geom.rec(*bbox)

        # PDE
        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = vpinn.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = vpinn.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = vpinn.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = vpinn.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = vpinn.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = vpinn.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = vpinn.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = vpinn.grad.jacobian(u, x, i=2, j=0)
            p_y = vpinn.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
            momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy))
            continuity = u_vel_x + v_vel_y

            return torch.cat([momentum_x, momentum_y, continuity], dim=1)
        self.load_ref_data(f'lid_driven_a{a}.dat')
        self.pde = ns_pde

        # bc
        def boundary_top(x):
            return torch.isclose(x[:,1], torch.full_like(x[:,1], bbox[3]))

        def boundary_side(x):
            return torch.logical_or(torch.isclose(x[:,0], torch.full_like(x[:,0], bbox[0])), torch.isclose(x[:,0], torch.full_like(x[:,0], bbox[1])))
        
        def boundary_bottom(x):
            return torch.isclose(x[:,1], torch.full_like(x[:, 1], bbox[2]))
        
        self.constrain = []
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: a * x[:, 0:1] * (1 - x[:, 0:1]),
                                                 100, boundary_top, u_component=0))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, boundary_top, u_component=1))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, boundary_side, u_component=0))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, boundary_side, u_component=1))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, boundary_bottom, u_component=2))
        
class NS_Back_Step(Problem):

    def __init__(
        self,
        nu=1 / 100,
        bbox=[0, 4, 0, 2],
        obstacle={
            'circ': [(1, 0.5, 0.2), (2, 0.5, 0.3), (2.7, 1.6, 0.2)],
            'rec': [(2.8, 3.6, 0.8, 1.1)]
        },
        # obstacle={},
        device='cpu'
    ):
        super().__init__(device)
        # output dim
        self.indim = 2
        self.outdim = 3
        # geom
        self.geom = vpinn.geom.rec(*bbox)
        self.bbox = bbox
        
        def boundary_in(x):
            return torch.logical_and(torch.isclose(x[:,0], torch.full_like(x[:,0], bbox[0])), x[:,1] < 1)

        def boundary_out(x):
            return torch.isclose(x[:,0], torch.full_like(x[:,0], bbox[1]))

        def boundary_other(x):
            return torch.logical_not(torch.logical_or(boundary_in(x), boundary_out(x)))
        
        def u_func(x):  # initial boundary flow rate
            return x[:, 1:2] * (1 - x[:, 1:2]) * 4
        
        self.constrain = []
        self.constrain.append(vpinn.bc.dirichlet(self.geom, u_func,
                                                 100, boundary_in, u_component=0))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]), 
                                                 100, boundary_in, u_component=1))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, boundary_out, 2))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, boundary_other, u_component=0))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, boundary_other, 1))
        line1 = vpinn.geom.line((0, 1), (2, 1))
        line2 = vpinn.geom.line((2, 1), (2, 2))
        self.constrain.append(vpinn.bc.dirichlet(line1, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, u_component=0 ))
        self.constrain.append(vpinn.bc.dirichlet(line1, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, u_component=1 ))
        self.constrain.append(vpinn.bc.dirichlet(line2, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, u_component=0 ))
        self.constrain.append(vpinn.bc.dirichlet(line2, lambda x: torch.zeros_like(x[:,0:1]),
                                                 100, u_component=1 ))
        
        for name, configs in obstacle.items():
            if name == 'circ':
                for c in configs:
                    diff_geom = vpinn.geom.circle(c[0], c[1], c[2])
                    self.constrain.append(vpinn.bc.dirichlet(diff_geom, lambda x: torch.zeros_like(x[:,0:1]), 100, u_component=0))
                    
            elif name == 'rec':
                for c in configs:
                    diff_geom = vpinn.geom.rec(c[0], c[1], c[2], c[3])
                    self.constrain.append(vpinn.bc.dirichlet(diff_geom, lambda x: torch.zeros_like(x[:,0:1]), 100, u_component=0))
        # PDE
        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = vpinn.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = vpinn.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = vpinn.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = vpinn.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = vpinn.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = vpinn.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = vpinn.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = vpinn.grad.jacobian(u, x, i=2, j=0)
            p_y = vpinn.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
            momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy))
            continuity = u_vel_x + v_vel_y

            return torch.cat([momentum_x, momentum_y, continuity], dim=1)

        self.pde = ns_pde

class NSEquation_Long(Problem):

    def __init__(self, nu=1 / 100, bbox=[0, 2, 0, 1, 0, 5], device='cpu'):
        super().__init__(device)
        # output dim
        self.indim = 3
        self.outdim = 3
        # geom
        self.bbox = bbox
        self.geomtime = vpinn.geomtime.timeplane(*bbox)

        # PDE
        COEF_A1 = 1
        COEF_A2 = 1
        COEF_A3 = 1

        def initial_f(x):
            return torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) * torch.sin(torch.pi * x[:, 2:3])

        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = vpinn.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = vpinn.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = vpinn.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = vpinn.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = vpinn.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = vpinn.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = vpinn.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = vpinn.grad.jacobian(u, x, i=2, j=0)
            p_y = vpinn.grad.jacobian(u, x, i=2, j=1)

            u_t = vpinn.grad.jacobian(u, x, i=0, j=2)
            v_t = vpinn.grad.jacobian(u, x, i=1, j=2)

            momentum_x = (u_t + u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy) + initial_f(x))
            momentum_y = (v_t + u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy) + initial_f(x))
            continuity = u_vel_x + v_vel_y

            return torch.cat([momentum_x, momentum_y, continuity], dim=1)

        self.pde = ns_pde

        # BC
        def boundary_in(x):
            return torch.isclose(x[:,0], torch.full_like(x[:, 0], bbox[0]))

        def boundary_out(x):
            return torch.isclose(x[:,0], torch.full_like(x[:, 1], bbox[1]))

        def boundary_other(x):
            return torch.logical_not(torch.logical_or(boundary_in(x), boundary_out(x)))

        def u_func(x):  # initial boundary flow rate
            y, t = x[:, 1:2], x[:, 2:3]
            return torch.sin(torch.pi * y) * (COEF_A1 * torch.sin(torch.pi * t) + COEF_A2 * torch.sin(3 * torch.pi * t) + COEF_A3 * torch.sin(5 * torch.pi * t))

        self.constrain = []
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, u_func, 100, boundary_in, u_component=0))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100, boundary_in, u_component=1))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100, boundary_out, u_component=2))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100, boundary_other, u_component=0))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100, boundary_other, u_component=1))
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, lambda x: torch.zeros_like(x), 100, u_component='ALL'))
        