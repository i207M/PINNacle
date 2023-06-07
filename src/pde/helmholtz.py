import numpy as np
import torch

import deepxde as dde
from . import baseclass


class Helmholtz2D(baseclass.BasePDE):

    def __init__(self, scale=1, A=(4, 4), k=1):
        super().__init__()
        # output dim
        self.output_dim = 1
        # Domain
        self.bbox = [0, scale, 0, scale]
        self.geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[scale, scale])

        # PDE
        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            def f(x):
                return torch.sin(A[0] * np.pi * x[:,0:1] / scale) * torch.sin(A[1] * np.pi * x[:, 1:2] / scale) * \
                        (k**2 - np.pi**2 * (A[0]**2 + A[1]**2) / scale**2)

            return u_xx + u_yy + k**2 * u - f(x)

        self.pde = pde
        self.set_pdeloss(num=1)

        def ref_sol(x):
            return np.sin(A[0] * np.pi * x[:, 0:1] / scale) * np.sin(A[1] * np.pi * x[:, 1:2] / scale)

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet',
        }])

        self.training_points()  # default
