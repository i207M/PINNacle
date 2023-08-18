import numpy as np
import vpinn
import torch
from .problem import Problem


class GrayScottEquation(Problem):

    def __init__(self, bbox=[-1, 1, -1, 1, 0, 200], b=0.04, d=0.1, epsilon=(1e-5, 5e-6), device='cpu'):
        super().__init__(device)
        # output dim
        self.indim = 3
        self.outdim = 2

        # geom
        self.bbox = bbox
        self.geomtime = vpinn.geomtime.timeplane(*bbox)

        # PDE
        def pde(x, y):
            u, v = y[:, 0:1], y[:, 1:2]

            u_t = vpinn.grad.jacobian(u, x, i=0, j=2)
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_yy = vpinn.grad.hessian(u, x, i=1, j=1)

            v_t = vpinn.grad.jacobian(v, x, i=0, j=2)
            v_xx = vpinn.grad.hessian(v, x, i=0, j=0)
            v_yy = vpinn.grad.hessian(v, x, i=1, j=1)
            return torch.cat([u_t - (epsilon[0] * (u_xx + u_yy) + b * (1 - u) - u * (v**2)), v_t - (epsilon[1] * (v_xx + v_yy) - d * v + u * (v**2))], dim=1)

        self.pde = pde

        # BC

        def ic_func(x):
            u1 = 1 - torch.exp(-80 * ((x[:, 0:1] + 0.05)**2 + (x[:, 1:2] + 0.02)**2))
            u2 = torch.exp(-80 * ((x[:, 0:1] - 0.05)**2 + (x[:, 1:2] - 0.02)**2))
            return torch.cat([u1, u2], dim=1)

        self.constrain = []
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, ic_func, 100, u_component=0))
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, ic_func, 100, u_component=1))


class KuramotoSivashinskyEquation(Problem):

    def __init__(self, bbox=[0, 2 * np.pi, 0, 1], alpha=100 / 16, beta=100 / (16 * 16), gamma=100 / (16**4), device='cpu'):
        super().__init__(device)
        # output dim
        self.indim = 2
        self.outdim = 1

        # geom
        self.bbox = bbox
        self.geomtime = vpinn.geomtime.timeline(*bbox)

        # PDE
        def pde(x, u):
            u_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u_t = vpinn.grad.jacobian(u, x, i=0, j=1)
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_xxxx = vpinn.grad.hessian(u_xx, x, i=0, j=0)

            return u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx

        self.pde = pde

        self.constrain = []
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, lambda x: torch.cos(x[:, 0:1]) * (1 + torch.sin(x[:, 0:1])), 100))
        