import pickle
import numpy as np
from scipy.interpolate import griddata
import scipy
import torch

import sys
sys.path.insert(0, '..')
import boundary_conditions
import losses
from problems import _Problem
sys.path.insert(0, './shared_modules')
from helper import Timer, cache_x


class WaveHetergeneous(_Problem):
    @property
    def name(self):
        return "WaveHetergeneous"

    def __init__(self):
        self.bbox = [-1, 1, -1, 1, 0, 5]
        self.d = (3, 1) # x,y,t
        self.load_ref_data("wave_darcy", timepde=(0, 5))
        self.num_js = 6
        self.darcy_2d_coef = np.loadtxt("../ref/darcy_2d_coef_256.dat")
        self.mu = (-0.5, 0)
        self.sigma = -0.3
    
    @cache_x()
    def coef(self, x):
        return torch.Tensor(
            scipy.interpolate.griddata(self.darcy_2d_coef[:, 0:2], self.darcy_2d_coef[:, 2], (x.detach().cpu().numpy()[:, 0:2]+1)/2)
        ).requires_grad_(False)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jt = j[:,0:1], j[:,1:2], j[:,2:3]
        jxx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jyy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        jtt = torch.autograd.grad(jt, x, torch.ones_like(jt), create_graph=True)[0][:,2:3]
        return y, jx, jy, jxx, jyy, jt, jtt
    
    def physics_loss(self, x, y, jx, jy, jxx, jyy, jt, jtt):
        physics = jxx + jyy - jtt / self.coef(x).to(x.device)
        return losses.l2_loss(physics, 0)

    def boundary_condition(self, x, y, jx, jy, jxx, jyy, jt, jtt, sd):
        # u(x,y,t) = u(x,y,0) + tanh(t)^2 NN(x,y,t)
        u0 = torch.exp( -( (x[:,0:1]-self.mu[0])**2 + (x[:,1:2]-self.mu[1])** 2) / (2*self.sigma**2) )
        u0x = (self.mu[0] - x[:,0:1]) / (2*self.sigma**2) * u0
        u0y = (self.mu[1] - x[:,1:2]) / (2*self.sigma**2) * u0
        u0xx = ( ((x[:,0:1]-self.mu[0])/self.sigma**2)**2 - 1/self.sigma**2 ) * u0
        u0yy = ( ((x[:,1:2]-self.mu[1])/self.sigma**2)**2 - 1/self.sigma**2 ) * u0
        t0, jt0, jjt0 = boundary_conditions.tanh2_2(x[:,2:3], 0, sd)
        y_new = u0 + t0*y
        jx_new = u0x + t0*jx
        jy_new = u0y + t0*jy
        jxx_new = u0xx + t0*jxx
        jyy_new = u0yy + t0*jyy
        jt_new = jt0*y + t0*jt
        jtt_new = jjt0*y + 2*jt0*jt + t0*jtt
        return y_new, jx_new, jy_new, jxx_new, jyy_new, jt_new, jtt_new

    def sample_bd(self, N_bd):
        nside = int(np.sqrt(N_bd//4))
        def mgrid(x1, x2, nx, y1, y2, ny, d3idx, d3val):
            xl, yl = np.linspace(x1, x2, nx), np.linspace(y1, y2, ny)
            xmesh, ymesh = np.meshgrid(*(xl, yl), indexing='ij')
            zmesh = np.ones_like(xmesh, dtype=xmesh.dtype) * d3val
            meshes = [xmesh, ymesh]
            meshes.insert(d3idx, zmesh)
            return np.stack(meshes, axis=-1).reshape(-1,3)
        bc_y0 = mgrid(-1, 1, nside, 0, 5, nside, 1, -1)
        bc_y1 = mgrid(-1, 1, nside, 0, 5, nside, 1, 1)
        bc_x0 = mgrid(-1, 1, nside, 0, 5, nside, 0, -1)
        bc_x1 = mgrid(-1, 1, nside, 0, 5, nside, 0, 1)
        bd_pts = np.concatenate([bc_y0,bc_y1,bc_x0,bc_x1], axis=0)
        return bd_pts
    
    def bd_loss(self, x, y, jx, jy, jxx, jyy, jt, jtt):
        # \partial u / \partial n = 0, x \in \partial \Omega
        x_cpu = x.detach().cpu()
        isx, isy = (torch.tensor(np.isclose(x_cpu[:,idim], -1) | np.isclose(x_cpu[:,idim], 1)) for idim in (0,1))
        xloss, yloss = (torch.where(isd.to(x.device), jd, 0.) for isd, jd in zip((isx, isy),(jx, jy)))
        return losses.l2_loss(xloss+yloss, 0)


class Wave2DLong(_Problem):
    @property
    def name(self):
        return "Wave2DLong"
    
    def __init__(self):
        self.bbox=[0, 1, 0, 1, 0, 100]
        self.d = (3,1)
        self.m1, self.m2 = 1, 1
        self.n1, self.n2 = 1, 1
        self.p1, self.p2 = 1, 1
        self.c1, self.c2 = 1, 1
        self.a = 20
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jt = j[:,0:1], j[:,1:2], j[:,2:3]
        jxx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jyy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        jtt = torch.autograd.grad(jt, x, torch.ones_like(jt), create_graph=True)[0][:,2:3]
        return y, jxx, jyy, jtt
    
    def physics_loss(self, x, y, jxx, jyy, jtt):
        return losses.l2_loss(jtt - (jxx + self.a**2 * jyy), 0)

    def exact_solution_(self, x):
        sol1 = self.c1 * np.sin(self.m1 * np.pi * x[:,0:1]) * np.sinh(self.n1 * np.pi * x[:,1:2]) * np.cos(self.p1 * np.pi * x[:,2:3])
        sol2 = self.c2 * np.sinh(self.m2 * np.pi * x[:,0:1]) * np.sin(self.n2 * np.pi * x[:,1:2]) * np.cos(self.p2 * np.pi * x[:,2:3])
        return sol1+sol2
    
    def exact_solution(self, x, batch_size):
        return (self.exact_solution_(x.cpu().detach()).to(x.device),) + (torch.ones((np.prod(batch_size),1), device=x.device),)*3
    
    def sample_bd(self, N_bd):
        nside = int(np.sqrt(N_bd//8))
        def mgrid(x1, x2, nx, y1, y2, ny, d3idx, d3val):
            xl, yl = np.linspace(x1, x2, nx), np.linspace(y1, y2, ny)
            xmesh, ymesh = np.meshgrid(*(xl, yl), indexing='ij')
            zmesh = np.ones_like(xmesh, dtype=xmesh.dtype) * d3val
            meshes = [xmesh, ymesh]
            meshes.insert(d3idx, zmesh)
            return np.stack(meshes, axis=-1).reshape(-1,3)
        ic = mgrid(0, 1, nside*2, 0, 1, nside*2, 2, 0)
        bc_y0 = mgrid(0, 1, nside, 0, 100, nside, 1, 0)
        bc_y1 = mgrid(0, 1, nside, 0, 100, nside, 1, 1)
        bc_x0 = mgrid(0, 1, nside, 0, 100, nside, 0, 0)
        bc_x1 = mgrid(0, 1, nside, 0, 100, nside, 0, 1)
        bd_pts = np.concatenate([ic,bc_y0,bc_y1,bc_x0,bc_x1], axis=0)
        #if not hasattr(self, "bd_cached"):
        #    self.bd_cached = self.exact_solution_(bd_pts)
        return bd_pts
    
    def bd_loss(self, x, y, jxx, jyy, jtt):
        return losses.l2_loss(y, torch.tensor(self.exact_solution_(x.cpu().detach()), device=y.device))


class WaveEquation1D(_Problem):
    @property
    def name(self):
        return "WaveEquation1D"
    
    def __init__(self):
        self.bbox=[0, 1, 0, 1]
        self.d = (2,1)
        self.m2 = 4
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jt = j[:,0:1], j[:,1:2]
        jxx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jtt = torch.autograd.grad(jt, x, torch.ones_like(jt), create_graph=True)[0][:,1:2]
        return y, jx, jt, jxx, jtt

    def physics_loss(self, x, y, jx, jt, jxx, jtt):
        physics = jtt - 4 * jxx
        return losses.l2_loss(physics, 0)

    def boundary_condition(self, x, y, jx, jt, jxx, jtt, sd):
        """
        Ansatz: u(x,t) = u(x,0) + NN(x,t)tanh(t)tanh(x)tanh(x-1)
        """
        t0, jt0, jjt0 = boundary_conditions.tanh_2(x[:,1:2], 0, sd)
        t1, t1x, t1xx = boundary_conditions.tanhtanh_2(x[:,0:1], 0, 1, sd)
        u0 = torch.sin(np.pi*x[:,0:1]) + 0.5*torch.sin(self.m2*np.pi*x[:,0:1])
        u0x = np.pi*torch.cos(np.pi*x[:,0:1]) + 0.5*self.m2*np.pi*torch.cos(self.m2*np.pi*x[:,0:1])
        u0xx = -np.pi**2*torch.sin(np.pi*x[:,0:1]) -0.5*self.m2**2*np.pi**2*torch.sin(self.m2*np.pi*x[:,0:1])
        y_new = u0 + y*t0*t1
        y_new_x = u0x + t0*(t1x*y+t1*jx)
        y_new_t = t1*(jt0*y+t0*jt)
        y_new_xx = u0xx + t0*(t1xx*y+2*t1x*jx+t1*jxx)
        y_new_tt = t1*(jjt0*y+2*jt0*jt+t0*jtt)
        return y_new, y_new_x, y_new_t, y_new_xx, y_new_tt
    
    def exact_solution(self, x, batch_size):
        y_exact = torch.sin(np.pi*x[:,0:1])*torch.cos(2*np.pi*x[:,1:2])+0.5*torch.sin(self.m2*np.pi*x[:,0:1])*torch.cos(2*self.m2*np.pi*x[:,1:2])
        y_exact.to(x.device)
        return (y_exact,) + (torch.ones((np.prod(batch_size),1), device=x.device),)*4