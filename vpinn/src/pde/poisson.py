import torch
import numpy as np
import vpinn
from .problem import Problem
from ..utils.func_cache import cache_tensor

class Poisson1(Problem):
    def __init__(self, bbox=[-1, 1, -1, 1], device='cpu'):
        super().__init__(device)
        self.indim = 2
        self.outdim = 1
        self.geom = vpinn.geom.rec(*bbox)
        self.u = u
        self.constrain = [vpinn.bc.dirichlet(self.geom, self.u, 100, locate_func=None, u_component=0)]
    
        def f(x, m=1, n=1, c=0.1, k=10):
            x = x[:, 0:1]
            y = x[:, 1:2]
            term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
                    (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
            term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
            return -(term1 + term2)

        def pde(x, u):    
            u_xx = vpinn.grad.grad(u, x, u_component=0, x_component=0, order=2)
            u_yy = vpinn.grad.grad(u, x, u_component=0, x_component=1, order=2)
            return u_xx + u_yy - f(x)
        
        self.pde = pde

        def u(x):
            x = x[:, 0:1]
            y = x[:, 1:2]
            return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)
        

class Poisson2d(Problem):
    def __init__(self, scale=1, device='cpu'):
        super().__init__(device)
        bbox=[-scale / 2, scale / 2, -scale / 2, scale / 2]
        self.indim = 2
        self.outdim = 1
        self.geom = vpinn.geom.rec(*bbox)
        self.constrain = [vpinn.bc.dirichlet(self.geom, lambda x: torch.ones_like(x[:,0:1]), 100, locate_func=None, u_component=0)]
        self.scale = scale

        def bc_circle():
            circles = [(0.3, 0.3, 0.1), (-0.3, 0.3, 0.1), (0.3, -0.3, 0.1), (-0.3, -0.3, 0.1)]
            geom = []
            bc_c = []
            for c in circles:
                geom.append(vpinn.geom.circle(c[0] * scale, c[1] * scale, c[2] * scale))
                bc_c.append(vpinn.bc.dirichlet(vpinn.geom.circle(c[0] * scale, c[1] * scale, c[2] * scale), 
                                            lambda x: torch.zeros_like(x[:,0:1]), 100))
            return bc_c
        self.constrain = self.constrain + bc_circle()

        
        def pde(x, u):    
            u_xx = vpinn.grad.grad(u, x, u_component=0, x_component=0, order=2)
            u_yy = vpinn.grad.grad(u, x, u_component=0, x_component=1, order=2)
            return u_xx + u_yy
        
        self.pde = pde
        self.load_ref_data('poisson1_cg_data.dat')

class Poisson_boltzmann2d(Problem):
    def __init__(self, k=8, mu=(1, 4), A=10, bbox=[-1, 1, -1, 1], circ=[(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, -0.7, 0.1), (-0.6, 0.5, 0.3)], device='cpu'):
        super().__init__(device)
        self.indim = 2
        self.outdim = 1
        self.geom = vpinn.geom.rec(*bbox)
        self.constrain = [vpinn.bc.dirichlet(self.geom, lambda x: torch.full_like(x[:,0:1], 0.2), 100, locate_func=None, u_component=0)]

        def bc_circle():
            geom = []
            bc_c = []
            for c in circ:
                geom.append(vpinn.geom.circle(c[0], c[1], c[2]))
                bc_c.append(vpinn.bc.dirichlet(vpinn.geom.circle(c[0], c[1], c[2]), 
                                            lambda x: torch.ones_like(x[:,0:1]), 1000))
            return bc_c
        self.constrain = self.constrain + bc_circle()

        def pde(x, u):    
            u_xx = vpinn.grad.grad(u, x, u_component=0, x_component=0, order=2)
            u_yy = vpinn.grad.grad(u, x, u_component=0, x_component=1, order=2)

            def f(xy):
                x, y = xy[:, 0:1], xy[:, 1:2]
                return A * (mu[0]**2 + x**2 + mu[1]**2 + y**2) \
                        * torch.sin(mu[0] * torch.pi * x) * torch.sin(mu[1] * torch.pi * y)

            return -(u_xx + u_yy) + k**2 * u - f(x)
        self.pde = pde

class Poisson3d(Problem):
    def __init__(
        self,
        bbox=[0, 1, 0, 1, 0, 1],
        interface_z=0.5,
        circ=[(0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2), (0.2, 0.8, 0.7, 0.1), (0.6, 0.2, 0.3, 0.1)],
        A=(20, 100),
        m=(1, 10, 5),
        k=(8, 10),
        mu=(1, 1),
        device='cpu'
    ):
        super().__init__(device)
        self.indim = 3
        self.outdim = 1
        self.geom = vpinn.geom.cube(*bbox)
        self.constrain = [vpinn.bc.neumann(self.geom, lambda x: torch.zeros_like(x[:,0:1]), 100, u_component=0)]
        self.sphere = circ
    
        def pde(x, u):
            
            u_xx = vpinn.grad.grad(u, x, u_component=0, x_component=0, order=2)
            u_yy = vpinn.grad.grad(u, x, u_component=0, x_component=1, order=2)
            u_zz = vpinn.grad.grad(u, x, u_component=0, x_component=2, order=2)
            
            def f(xyz):
                x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
                xlen2 = x**2 + y**2 + z**2
                part1 = torch.exp(torch.sin(m[0] * torch.pi * x) + torch.sin(m[1] * torch.pi * y) + torch.sin(m[2] * torch.pi * z)) * (xlen2 - 1) / (xlen2 + 1)
                part2 = torch.sin(m[0] * torch.pi * x) * torch.sin(m[1] * torch.pi * y) * torch.sin(m[2] * torch.pi * z)
                return A[0] * part1 + A[1] * part2
            
            mus = torch.where(x[:, 2] < interface_z, mu[0], mu[1]).unsqueeze(dim=-1)
            ks = torch.where(x[:, 2] < interface_z, k[0]**2, k[1]**2).unsqueeze(dim=-1)
            return -mus * (u_xx + u_yy + u_zz) + ks * u - f(x)
        self.pde = pde

        def bc_interface(z):
            interface = vpinn.geom.cube(bbox[0], bbox[1], bbox[2], bbox[3], z, z)
            bc = vpinn.bc.neumann(interface, lambda x: torch.zeros_like(x[:,0:1]), 100, u_component=0)
            return bc
        self.constrain.append(bc_interface(0.5))

        def griddata_func():
            self.layers = 5
            x = torch.linspace(0, 1, 20 * self.layers + 1).to(self.device)
            y = x
            z = x
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            xx = xx.reshape(-1,)
            yy = yy.reshape(-1,)
            zz = zz.reshape(-1,)
            
            sphere = torch.tensor(self.sphere).to(self.device)
            mask = torch.ones(xx.shape[0], dtype=bool, device=self.device)
            for j in range(len(sphere)):
                mask &= ((xx - sphere[j, 0]) ** 2 + (yy - sphere[j, 1]) ** 2 + (zz - sphere[j, 2]) ** 2 >= sphere[j, 3] ** 2)

            xx = xx[mask].reshape(-1, 1)
            yy = yy[mask].reshape(-1, 1)
            zz = zz[mask].reshape(-1, 1)

            return torch.cat([xx, yy, zz], dim=1).to('cpu')

        self.griddata = griddata_func()


class Poisson2d_Many_subdomains(Problem):
    def __init__(self, bbox=[-10, 10, -10, 10], split=(5, 5), freq=2, device='cpu'):
        super().__init__(device)
        self.indim = 2
        self.outdim = 1
        self.geom = vpinn.geom.rec(*bbox)

        self.a_cof = np.loadtxt("ref/poisson_a_coef.dat")
        self.f_cof = np.loadtxt("ref/poisson_f_coef.dat").reshape(split[0], split[1], freq, freq)
        block_size = np.array([(bbox[1] - bbox[0] + 2e-5) / split[0], (bbox[3] - bbox[2] + 2e-5) / split[1]])

        def domain(x):
                reduced_x = (x - np.array(bbox[::2]) + 1e-5)
                dom = np.floor(reduced_x / block_size).astype("int32")
                return dom, reduced_x - dom * block_size

        def a(x):
            dom, _ = domain(x)
            return self.a_cof[dom[0], dom[1]]

        a = np.vectorize(a, signature="(2)->()")

        def f(x):
            dom, res = domain(x)
            def f_fn(coef):
                ans = coef[0, 0]
                for i in range(coef.shape[0]):
                    for j in range(coef.shape[1]):
                        tmp = np.sin(np.pi * np.array((i, j)) * (res / block_size))
                        ans += coef[i, j] * tmp[0] * tmp[1]
                return ans

            return f_fn(self.f_cof[dom[0], dom[1]])
        
        f = np.vectorize(f, signature="(2)->()")

        @cache_tensor
        def get_coef(x):
            x = x.detach().cpu()
            return torch.Tensor(a(x)).unsqueeze(dim=-1).to(self.device), torch.Tensor(f(x)).unsqueeze(dim=-1).to(self.device)
        
        def pde(x, u):
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            u_yy = vpinn.grad.hessian(u, x, i=1, j=1)

            a, f = get_coef(x)
            return a * (u_xx + u_yy) + f
        
        self.pde = pde
        self.constrain = [vpinn.bc.robin(self.geom, lambda x, y: -y, 100)]
        