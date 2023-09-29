import torch
import vpinn

from .problem import Problem


class PoissonInv(Problem):

    def __init__(self, device='cpu'):
        super().__init__(device)
        # output dim
        self.indim = 2
        self.outdim = 1
        # geom
        self.bbox = [0, 1] * 2
        self.geom = vpinn.geom.rec(*self.bbox)

        # reference
        def a_ref(xy):
            x, y = xy[:, 0:1], xy[:, 1:2]
            return 1 / (1 + x**2 + y**2 + (x - 1)**2 + (y - 1)**2)

        def u_ref(xy):
            x, y = xy[:, 0:1], xy[:, 1:2]
            return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) + torch.randn(x.shape).to(device) * 0.1
        
        # PDE
        def pde(x, u, a):
            u_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u_y = vpinn.grad.jacobian(u, x, i=0, j=1)
            d_au = vpinn.grad.jacobian(a * u_x, x, i=0, j=0) + \
                   vpinn.grad.jacobian(a * u_y, x, i=0, j=1)

            def f(xy):
                x, y = xy[:, 0:1], xy[:, 1:2]
                return 2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * a_ref(xy) + \
                    2 * torch.pi * ((2*x+1) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y) + (2*y+1) * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)) * a_ref(xy)**2

            return d_au + f(x)

        self.pde = pde

        self.constrain = []
        self.constrain.append(vpinn.bc.dirichlet(self.geom, a_ref, 100))
        self.constrain.append(vpinn.bc.dirichlet(self.geom, u_ref, 100, inverse=True))
        
        self.inverse = True
        self.u_ref = u_ref
        self.a_ref = a_ref
        

class HeatInv(Problem):

    def __init__(self, device='cpu'):
        super().__init__(device)
        # output dim
        self.outdim = 3
        self.outdim = 1
        # geom
        self.bbox = [-1, 1] * 2 + [0, 1]
        self.geomtime = vpinn.geomtime.timeplane(*self.bbox)

        # PDE
        def pde(x, u, a):
            u_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u_y = vpinn.grad.jacobian(u, x, i=0, j=1)
            u_t = vpinn.grad.jacobian(u, x, i=0, j=2)
            d_au = vpinn.grad.jacobian(a * u_x, x, i=0, j=0) + \
                   vpinn.grad.jacobian(a * u_y, x, i=0, j=1)

            def f(xyt):
                x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
                s, c, p = torch.sin, torch.cos, torch.pi
                return torch.exp(-t) * ((4*p**2-1) * s(p*x) * s(p*y) + \
                    p**2 * (2 * s(p*x)**2 * s(p*y)**2 - c(p*x)**2 * s(p*y)**2 - s(p*x)**2 * c(p*y)**2))

            return u_t - d_au - f(x)

        self.pde = pde

        def u_ref(xyt):
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            return torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) + torch.randn(x.shape).to(device) * 0.1

        def a_ref(xyt):  # irrelevent to time domain
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            return 2 + torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        
        self.a_ref = a_ref
        self.u_ref = u_ref
        self.inverse = True

        self.constrain = []
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, a_ref, 100))
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, a_ref, 100))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, u_ref, 100, inverse=False))
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, u_ref, 100, inverse=False))