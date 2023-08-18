import torch
import vpinn
import scipy
import numpy as np
from .problem import Problem

class Burgers1d(Problem):
    def __init__(self, geom=[-1, 1], time=[0, 1], nu=0.01 / np.pi, device='cpu'):
        super().__init__(device)
        self.indim = 2
        self.outdim = 1
        self.geom = vpinn.geomtime.timeline(*(geom + time))
        x1 = geom[0]
        x2 = geom[1]
        self.bc = vpinn.bc.dirichlet(geometry=self.geom, 
                                    func=lambda x: torch.zeros_like(x[:,0:1]), 
                                    num=100,
                                    locate_func=lambda x: torch.isclose(x[:, 0], torch.full_like(x[:, 0], x1)),
                                    u_component=0)
        
        self.bc = vpinn.bc.dirichlet(geometry=self.geom, 
                                    func=lambda x: torch.zeros_like(x[:,0:1]),
                                    num=100,
                                    locate_func=lambda x: torch.isclose(x[:, 0], torch.full_like(x[:, 0], x2)),
                                    u_component=0)
        
        self.ic = vpinn.ic.dirichlet(domain=self.geom, 
                                     func=lambda x: -torch.sin(torch.pi * x[:,0:1]), 
                                     num=100, 
                                     locate_func=None,
                                     u_component=0)
        self.constrain = [self.bc, self.ic]

        def pde(x, u):    
            u_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u_t = vpinn.grad.jacobian(u, x, i=0, j=1)
            u_xx = vpinn.grad.hessian(u, x, i=0, j=0)
            return u_t + u * u_x - nu * u_xx
        
        self.pde = pde

class Burgers2d(Problem):
    def __init__(self, bbox=[0, 4, 0, 4, 0, 1], nu=0.001, L=4, T=1, M=1, device='cpu', data_id=0):
        super().__init__(device)
        self.indim = 3
        self.outdim = 2
        x1, x2, y1, y2, t = (0, L, 0, L, T)
        self.geom = vpinn.geomtime.timeplane(*bbox)
        self.bc1 = vpinn.bc.periodic(geometry=self.geom, 
                                    func=None, 
                                    num=100,
                                    locate_func=[lambda x: torch.isclose(x[:, 0], torch.full_like(x[:, 0], x1)), 
                                                 lambda x: torch.isclose(x[:, 0], torch.full_like(x[:, 0], x2))],
                                    u_component='ALL')
        
        self.bc2 = vpinn.bc.periodic(geometry=self.geom, 
                                    func=None, 
                                    num=100,
                                    locate_func=[lambda x: torch.isclose(x[:, 1], torch.full_like(x[:, 1], y1)), 
                                                 lambda x: torch.isclose(x[:, 1], torch.full_like(x[:, 1], y2))],
                                    u_component='ALL')

        # self.ic_coefs = np.loadtxt("ref/burgers2d_coef.dat")
        path = "ref/burgers2d_init_"
        self.ics = (np.loadtxt(path + 'u_' + f'{data_id}' + '.dat'), np.loadtxt(path + 'v_' + f'{data_id}' + '.dat'))
        
        self.interpolators = [scipy.interpolate.LinearNDInterpolator(ic[:, :2], ic[:, 2:]) for ic in self.ics]
        
        def ic_func(x, component):
            x = x.cpu().detach().numpy()
            interpolated_data = self.interpolators[component](x[:, :2])
            return torch.from_numpy(interpolated_data).to(torch.float32).to(self.device)
    
        self.ic1 = vpinn.ic.dirichlet(domain=self.geom, 
                                     func=lambda x: torch.cat([ic_func(x, 0), ic_func(x, 1)], dim=1), 
                                     num=100, 
                                     locate_func=None,
                                     u_component='ALL')
        
        self.constrain = [self.bc1, self.bc2, self.ic1]

        self.load_ref_data(f'burgers2d_{data_id}.dat')
        # def ic_func(x, component):
        #     x = x.cpu().detach().numpy()
        #     A = self.ic_coefs[:2 * (2 * L + 1)**2].reshape(2, 2 * L + 1, 2 * L + 1)
        #     B = self.ic_coefs[2 * (2 * L + 1)**2:4 * (2 * L + 1)**2].reshape(2, 2 * L + 1, 2 * L + 1)
        #     C = self.ic_coefs[4 * (2 * L + 1)**2:]

        #     w = np.zeros((x.shape[0], 1))
        #     for i in range(-L, L + 1):
        #         for j in range(-L, L + 1):
        #             w += A[component][i][j] * np.sin(2 * np.pi * (i * x[:, 0:1] + j * x[:, 1:2])) \
        #                 + B[component][i][j] * np.cos(2 * np.pi * (i * x[:, 0:1] + j * x[:, 1:2]))

        #     return torch.from_numpy(2 * w / M + C[component]).reshape(-1, 1).to(torch.float32).to(self.device)
    
        def pde(x, u):
            u1_t = vpinn.grad.jacobian(u, x, i=0, j=2)
            u2_t = vpinn.grad.jacobian(u, x, i=1, j=2)
            
            u1_x = vpinn.grad.jacobian(u, x, i=0, j=0)
            u2_x = vpinn.grad.jacobian(u, x, i=1, j=0)
            
            u1_y = vpinn.grad.jacobian(u, x, i=0, j=1)
            u2_y = vpinn.grad.jacobian(u, x, i=1, j=1)
            
            u1_xx = vpinn.grad.hessian(u, x, i=0, j=0, component=0)
            u1_yy = vpinn.grad.hessian(u, x, i=1, j=1, component=0)
            
            u2_xx = vpinn.grad.hessian(u, x, i=0, j=0, component=1)
            u2_yy = vpinn.grad.hessian(u, x, i=1, j=1, component=1)
            v = nu
            
            return torch.cat([u1_t + u[:,0:1] * u1_x + u[:,1:2] * u1_y - v * (u1_xx + u1_yy),
                        u2_t + u[:,0:1] * u2_x + u[:,1:2] * u2_y - v * (u2_xx + u2_yy)], dim=1)
        self.pde = pde
        