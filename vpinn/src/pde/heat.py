import torch
import vpinn
import numpy as np
from .problem import Problem
from ..utils.func_cache import cache_tensor
from scipy import interpolate

class Heat2d_Varying_Source(Problem):

    def __init__(self, bbox=[0, 1, 0, 1, 0, 5], A=200, m=(1, 5, 1), device='cpu'):
        super().__init__(device)
        self.bbox=bbox
        self.indim = 3
        self.outdim = 1
        self.geomtime = vpinn.geomtime.timeplane(*bbox)

        # PDE
        self.heat_2d_coef = np.loadtxt("ref/heat_2d_coef_256.dat")

        @cache_tensor
        def coef(x):
            return torch.Tensor(
                interpolate.griddata(self.heat_2d_coef[:, 0:2], self.heat_2d_coef[:, 2], x.detach().cpu().numpy()[:, 0:2], method='nearest')
            ).unsqueeze(dim=1).to(self.device)

        def heat_pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0) + vpinn.grad.hessian(u, x, i=1, j=1)
            u_t = vpinn.grad.jacobian(u, x, i=0, j=2)

            def f(x):
                return A * torch.sin(m[0] * torch.pi * x[:, 0:1]) * torch.sin(m[1] * torch.pi * x[:, 1:2]) * torch.sin(m[2] * torch.pi * x[:, 2:3])

            return u_t - coef(x) * u_xx - f(x)

        self.pde = heat_pde

        self.constrain = [vpinn.ic.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100),
                   vpinn.bc.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100)]
    
class Heat_Multi_scale(Problem):

    def __init__(self, bbox=[0, 1, 0, 1, 0, 5], device='cpu', a=20):
        super().__init__(device)
        self.indim = 3
        self.outdim = 1
        self.bbox = bbox
        self.geomtime = vpinn.geomtime.timeplane(*bbox)
        
        def pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_yy = vpinn.grad.hessian(u, x, i=1, j=1)
            u_t = vpinn.grad.jacobian(u, x, j=2)
            return u_t - pde_coef[0] * u_xx - pde_coef[1] * u_yy
        
        self.pde = pde

        # BCs
        # def f_func(x):
        #     return torch.sin(self.INITIAL_COEF_1 * x[:, 0:1]) * torch.sin(self.INITIAL_COEF_2 * x[:, 1:2])
        init_coef=(a * np.pi, np.pi)
        pde_coef=(1 / np.square(500 * np.pi), 1 / np.square(np.pi))
        
        def ref_sol(xt):
            return torch.sin(init_coef[0] * xt[:, 0:1]) * torch.sin(init_coef[1] * xt[:, 1:2]) * \
                   torch.exp(-(pde_coef[0]*init_coef[0]**2 + pde_coef[1]*init_coef[1]**2) * xt[:, 2:3])

        self.u_ref = ref_sol
        self.constrain = []
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, ref_sol, 100))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100))
        
class HeatComplex(Problem):

    def __init__(self, bbox=[-8, 8, -12, 12, 0, 3], device='cpu'):
        super().__init__(device)
        self.indim = 3
        self.outdim = 1
        self.geomtime= vpinn.geomtime.timeplane(*bbox)
        self.bbox = bbox
        
        def pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_yy = vpinn.grad.hessian(u, x, i=1, j=1)
            u_t = vpinn.grad.jacobian(u, x, j=2)

            return u_t - u_xx - u_yy

        self.pde = pde
        self.constrain = []
        self.constrain.append(vpinn.bc.robin(self.geomtime, lambda x, u: 0.1 - u, 50))
        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 50))
        # big circles
        big_centers = [(-4, -3), (4, -3), (-4, 3), (4, 3), (-4, -9), (4, -9), (-4, 9), (4, 9), (0, 0), (0, 6), (0, -6)]
        for center in big_centers:
            big_circle = vpinn.geom.disk(center[0], center[1], bbox[-1], 1)
        self.constrain.append(vpinn.bc.robin(big_circle, lambda x, u: 5 - u, 50))
            
        # small circles
        small_centers = [(-3.2, -6), (-3.2, 6), (3.2, -6), (3.2, 6), (-3.2, 0), (3.2, 0)]
        for center in small_centers:
            small_circle = vpinn.geom.disk(center[0], center[1], bbox[-1], 0.4)
            self.constrain.append(vpinn.bc.robin(small_circle, lambda x, u: 1 - u, 50))
        
class HeatLongTime(Problem):

    def __init__(self, bbox=[0, 1, 0, 1, 0, 100], k=1, m1=4, m2=2, device='cpu'):
        super().__init__(device)
        self.indim = 3
        self.outdim = 1
        # geom
        self.bbox = bbox
        self.geomtime = vpinn.geomtime.timeplane(*bbox)

        # pde
        INITIAL_COEF_1 = 4 * np.pi
        INITIAL_COEF_2 = 3 * np.pi

        def pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_yy = vpinn.grad.hessian(u, x, i=1, j=1)
            u_t = vpinn.grad.jacobian(u, x, j=2)

            return u_t - 0.001 * (u_xx + u_yy) - 5 * torch.sin(k * torch.square(u)) * \
                (1 + 2 * torch.sin(x[:, 2:3] * np.pi / 4)) * torch.sin(m1 * np.pi * x[:, 0:1]) * torch.sin(m2 * np.pi * x[:, 1:2])
            

        self.pde = pde

        self.constrain = []
        
        # BCs
        def f_func(x):
            return torch.sin(INITIAL_COEF_1 * x[:, 0:1]) * torch.sin(INITIAL_COEF_2 * x[:, 1:2])

        self.constrain.append(vpinn.ic.dirichlet(self.geomtime, f_func, 100))
        self.constrain.append(vpinn.bc.dirichlet(self.geomtime, lambda x: torch.zeros_like(x[:,0:1]), 100))
