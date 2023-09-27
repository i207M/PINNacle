import torch
import vpinn
import numpy as np
from .problem import Problem
from ..utils.func_cache import cache_tensor
from scipy import interpolate

class WaveEquation1D(Problem):

    def __init__(self, C=2, scale=1, a=2, device='cpu'):
        super().__init__(device)
        # output dim
        self.indim = 2
        self.outdim = 1
        # geom
        self.bbox = [0, scale, 0, scale]
        self.geomtime = vpinn.geomtime.timeline(*self.bbox)

        # define wave equation
        def wave_pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_tt = vpinn.grad.hessian(u, x, i=1, j=1)

            return u_tt - C**2 * u_xx

        self.pde = wave_pde

        def ref_sol(x):
            x = x / scale
            return (torch.sin(torch.pi * x[:, 0:1]) * torch.cos(2 * torch.pi * x[:, 1:2]) + 0.5 * torch.sin(a * torch.pi * x[:, 0:1]) * torch.cos(2 * a * torch.pi * x[:, 1:2]))

        self.u_ref = ref_sol

        def boundary_x0(x):
            return torch.isclose(x[:,0], torch.full_like(x[:,0], self.bbox[0])) |  torch.isclose(x[:,0], torch.full_like(x[:,0], self.bbox[1]))

        def boundary_t0(x):
            return torch.isclose(x[:,1], torch.full_like(x[:,1], self.bbox[2]))
        
        self.constrain = []
        self.constrain.append(vpinn.ic.neumann(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100, boundary_t0))
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, ref_sol, 100, boundary_t0))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, ref_sol, 100, boundary_x0))

class WaveHeterogeneous(Problem):

    def __init__(self, bbox=[-1, 1, -1, 1, 0, 5], mu=(-0.5, 0), sigma=0.3, device='cpu'):
        super().__init__(device)
        # output dim
        self.indim = 3
        self.outdim = 1
        # geom
        # NOTE: no circ are deleted, since the pde is currently not regraded as TimePDE and 3D-CSGDifference is difficult)
        self.bbox = bbox
        self.geomtime = vpinn.geomtime.timeplane(*bbox)

        # PDE
        self.darcy_2d_coef = np.loadtxt("ref/darcy_2d_coef_256.dat")

        @cache_tensor
        def coef(x):
            return torch.Tensor(
                interpolate.griddata(self.darcy_2d_coef[:, 0:2], self.darcy_2d_coef[:, 2], (x.detach().cpu().numpy()[:, 0:2] + 1) / 2)
            ).unsqueeze(dim=-1)

        def wave_pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0) + vpinn.grad.hessian(u, x, i=1, j=1)
            u_tt = vpinn.grad.hessian(u, x, i=2, j=2)

            return u_xx - u_tt / coef(x).to(device)

        self.pde = wave_pde

        # BCs
        # def boundary_t0(x):
        #     return torch.isclose(x[:,2], torch.full_like(x[:, 2], bbox[4]))
        
        # def boundary_rec(x):
        #     return ((torch.isclose(x[:, 0], torch.full_like(x[:, 0], bbox[0])) | torch.isclose(x[:, 0], torch.full_like(x[:, 0], bbox[1])) | torch.isclose(x[:, 1], torch.full_like(x[:,1], bbox[2])) | torch.isclose(x[:, 1], torch.full_like(x[:,1],bbox[3])))).reshape(-1, 1)

        def initial_condition(x):
            return torch.exp(-((x[:, 0:1] - mu[0])**2 + (x[:, 1:2] - mu[1])**2) / (2 * sigma**2))

        self.constrain = []
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, initial_condition, 100))
        self.constrain.append(vpinn.ic.neumann(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100))
        self.constrain.append(vpinn.bc.neumann(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100))
        
class WaveEquation2D_Long(Problem):

    def __init__(self, bbox=[0, 1, 0, 1, 0, 100], a=np.sqrt(2), m1=1, m2=3, n1=1, n2=2, p1=1, p2=1, device='cpu'):
        super().__init__(device)

        # output dim
        self.indim = 3
        self.outdim = 1
        # geom
        self.bbox = bbox
        self.geomtime = vpinn.geomtime.timeplane(*bbox)

        # pde
        INITIAL_COEF_1 = 1
        INITIAL_COEF_2 = 1

        def pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_yy = vpinn.grad.hessian(u, x, i=1, j=1)
            u_tt = vpinn.grad.hessian(u, x, i=2, j=2)

            return u_tt - (u_xx + a * a * u_yy)
        self.pde = pde

        # BCs
        def ref_sol(x):
            return (
                INITIAL_COEF_1 * torch.sin(m1 * torch.pi * x[:, 0:1]) * torch.sinh(n1 * torch.pi * x[:, 1:2]) * torch.cos(p1 * torch.pi * x[:, 2:3])
                + INITIAL_COEF_2 * torch.sinh(m2 * torch.pi * x[:, 0:1]) * torch.sin(n2 * torch.pi * x[:, 1:2]) * torch.cos(p2 * torch.pi * x[:, 2:3])
            )

        self.u_ref = ref_sol

        self.constrain = []
        self.constrain.append(vpinn.ic.neumann(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, ref_sol, 100))
        