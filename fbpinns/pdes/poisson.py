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


class Poisson2D_1(_Problem):
    """
    Solves the time-independent 2D Poisson equation
    - \Laplace u = f(x, y)
    where f is a given complicated term of x,y,c,m,k,n

    for -1.0 <= x, y <= +1.0

    Boundary conditions:
    u(x,-1) = 0, u(x,1) = 0
    u(-1,y) = tanh(-k)sin(2\pi ny), u(1,y) = tanh(k)sin(2\pi ny)
    """

    @property
    def name(self):
        return "Poisson2D_1"
    
    def __init__(self, c=0.1, m=0.5, k=0, n=0.5):
        self.bbox = [-1, 1, -1, 1]
        self.c = c
        self.m = m
        self.k = k
        self.n = n
        # dimensionality of x and y
        self.d = (2,1)
    
    def comp_u(self, x):
        # u0 = c*sin(2\pi mx)+tanh(kx)
        # u1 = sin(2\pi ny)
        u0 = self.c * torch.sin(2*np.pi * self.m * x[:,0:1]) + torch.tanh(self.k * x[:,0:1])
        u1 = torch.sin(2*np.pi * self.n * x[:,1:2])
        return u0, u1

    def comp_ujj(self, x):
        ujj0 = - 4*np.pi**2 * self.m**2 * self.c * torch.sin(2*np.pi * self.m * x[:,0:1]) \
                - 2*self.k**2 * torch.tanh(self.k * x[:,0:1]) * (1-torch.tanh(self.k * x[:,0:1])**2)
        ujj1 = - 4*np.pi**2 * self.n**2 * torch.sin(2*np.pi * self.n * x[:,1:2])
        return ujj0, ujj1

    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        # 是用main中PINNS的训练流程调试的，没有管FBPINNS的流程
        # x[:,0] vs x[:,0:1] 导致不同的结果，所以不能直接用comp_u, comp_ujj
        u0_prime = self.c * torch.sin(2*np.pi * self.m * x[:,0]) + torch.tanh(self.k * x[:,0])
        u1_prime = torch.sin(2*np.pi * self.n * x[:,1])
        ujj0_prime = - 4*np.pi**2 * self.m**2 * self.c * torch.sin(2*np.pi * self.m * x[:,0]) \
                - 2*self.k**2 * torch.tanh(self.k * x[:,0]) * (1-torch.tanh(self.k * x[:,0])**2)
        ujj1_prime = - 4*np.pi**2 * self.n**2 * torch.sin(2*np.pi * self.n * x[:,1])
        physics_prime = jj0[:,0] + jj1[:,0] - ujj0_prime*u1_prime - u0_prime*ujj1_prime
        return losses.l2_loss(physics_prime, 0)
    
    def get_gradients(self, x, y):
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        jj0 = torch.autograd.grad(j0, x, torch.ones_like(j0), create_graph=True)[0][:,0:1]
        jj1 = torch.autograd.grad(j1, x, torch.ones_like(j1), create_graph=True)[0][:,1:2]
        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        # 'x' is x[:,0:1], 'y' is x[:,1:2] , 'u' is y
        # Apply u(x,y) = tanh(x-1)*tanh(x+1)*tanh(y-1)*tanh(y+1)*NN(x,y) + sin(2\pi ny)tanh(kx) ansatz
        t0, jt0, jjt0 = boundary_conditions.tanhtanh_2(x[:,0:1], -1, 1, sd)
        t1, jt1, jjt1 = boundary_conditions.tanhtanh_2(x[:,1:2], -1, 1, sd)
        tan = torch.tanh(self.k*x[:,0:1])
        tan_j = self.k*(1-torch.tanh(self.k*x[:,0:1])**2)
        tan_jj = -2*self.k**2*torch.tanh(self.k*x[:,0:1])*(1-torch.tanh(self.k*x[:,0:1])**2)
        sin = torch.sin(2*np.pi*self.n*x[:,1:2])
        sin_j = 2*np.pi*self.n * torch.cos(2*np.pi*self.n*x[:,1:2])
        sin_jj = -4*np.pi**2*self.n**2 * torch.sin(2*np.pi*self.n*x[:,1:2])
        y_new = t0*t1*y + tan*sin
        j0_new = jt0*t1*y + t0*t1*j0 + tan_j*sin
        j1_new = t0*jt1*y + t0*t1*j1 + tan*sin_j
        jj0_new = t1*(jjt0*y + 2*jt0*j0 + t0*jj0) + tan_jj*sin
        jj1_new = t0*(jjt1*y + 2*jt1*j1 + t1*jj1) + tan*sin_jj
        return y_new, j0_new, j1_new, jj0_new, jj1_new

    def exact_solution(self, x, batch_size):
        # u = u0 * u1 = ( c*sin(2\pi mx)+tanh(kx) )*sin(2\pi ny)
        u0, u1 = self.comp_u(x)
        uj0 = self.c * 2*np.pi * self.m * torch.cos(2*np.pi * self.m * x[:,0:1])\
              + self.k * (1-torch.tanh(self.k * x[:,0:1])**2)
        uj1 = 2*np.pi * self.n * torch.cos(2*np.pi * self.n * x[:,1:2])
        ujj0, ujj1 = self.comp_ujj(x)
        return u0*u1, uj0*u1, u0*uj1, ujj0*u1, u0*ujj1


class Poisson2D_Hole(_Problem):
    """
    Solves the Poisson-Boltzmann 2d irregular domain
    """
    @property
    def name(self):
        return "Poisson2D_Hole"

    def mask_x(self, x):
        """
        Input a torch.tensor of shape (b,nd), Output a boolean torch.tensor of shape (nd)
        """
        masks = [None] * len(self.filterparams)
        for i, param in enumerate(self.filterparams):
            masks[i] = torch.sum( (x - torch.tensor(param[:2], device=x.device))**2, dim=1) > param[2]**2
        for i in range(1,len(self.filterparams)):
            masks[i] = masks[i] & masks[i-1]
        return masks[-1]
    
    def sample_bd(self, N_bd):
        """
        Sample (approximately) N_bd points on the boundary
        """
        pts = []
        N_bd = (N_bd + 7) // 8
        for i in range(N_bd):
            pts.append([-1, -1+2*i/N_bd])
            pts.append([-1+2*i/N_bd, 1])
            pts.append([1, 1-2*i/N_bd])
            pts.append([1-2*i/N_bd, -1])
        for i in range(N_bd):
            theta = 2*np.pi*i/N_bd
            for param in self.filterparams:
                pts.append([param[0]+param[2]*np.cos(theta), param[1]+param[2]*np.sin(theta)])
        return pts

    def __init__(self):
        # dimensionality of x and y
        self.bbox=[-1, 1, -1, 1]
        self.d = (2,1)
        self.filterparams = [(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, -0.7, 0.1), (-0.6, 0.5, 0.3)]
        self.mu1 = 1
        self.mu2 = 4
        self.k = 8
        self.A = 10
        self.load_ref_data("poisson_boltzmann2d")
    
    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        f = self.A * (self.mu1**2 + self.mu2**2 + x[:,0]**2 + x[:,1]**2) * \
                torch.sin(self.mu1*np.pi*x[:,0]) * torch.sin(self.mu2*np.pi*x[:,1])
        physics = jj0[:,0] + jj1[:,0] - self.k**2 * y[:,0] + f
        return losses.l2_loss(physics, 0)

    def bd_loss(self, x, y, j0, j1, jj0, jj1):
        is4sides = np.logical_or( np.isclose(np.abs(x[:,0].detach().cpu()), 1) , np.isclose(np.abs(x[:,1].detach().cpu()), 1) )
        is4sides = torch.tensor(is4sides, device=x.device)
        bd_true = torch.where(is4sides, 0.2, 1.)
        return losses.l2_loss(y[:,0],bd_true)

    def get_gradients(self, x, y):
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        jj0 = torch.autograd.grad(j0, x, torch.ones_like(j0), create_graph=True)[0][:,0:1]
        jj1 = torch.autograd.grad(j1, x, torch.ones_like(j1), create_graph=True)[0][:,1:2]
        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        # do not use ansatz
        return y, j0, j1, jj0, jj1

    def exact_solution(self, x, batch_size):
        # return value is used for visualization only, the nan values produced by griddate (out of convex hull) leads to blanks in the imshow (this is expected behavior)
        x_lims = [ (x[:,in_dim].cpu().min(), x[:,in_dim].cpu().max()) for in_dim in range(self.d[0])]
        x_mesh = [np.linspace(lim[0], lim[1], b) for lim,b in zip(x_lims, batch_size)]
        grid_x_tup = np.meshgrid(*x_mesh, indexing="ij")
        grid_interp = griddata(self.ref_x, self.ref_y, tuple(grid_x_tup), method="cubic") # should change code if ref_y is multidimentional
        return (torch.tensor(grid_interp.reshape(-1, 1), device=x.device),) + (torch.ones((np.prod(batch_size),1), device=x.device),)*4


class Poisson2D_Classic(_Problem):
    """
    Solves the Classic \delta u = 0 equation on irregular domain
    """
    @property
    def name(self):
        return "Poisson2D_Classic"
    
    def __init__(self, xscale=1):
        # dimensionality of x and y
        self.bbox=[-0.5, 0.5, -0.5, 0.5]
        self.d = (2,1)
        self.filterparams = [(0.3, 0.3, 0.1), (-0.3, 0.3, 0.1), (0.3, -0.3, 0.1), (-0.3, -0.3, 0.1)]
        self.load_ref_data("poisson1_cg_data")
        # scale the input dimensions, this does not change the solution
        self.bbox = [x*xscale for x in self.bbox]
        self.filterparams = [tuple(x*xscale for x in tup) for tup in self.filterparams]
        self.ref_x *= xscale # self.ref_data changes after this operation. Do not tranform twice.
        self.xscale = xscale
        self.num_js = 4

    def mask_x(self, x):
        """
        Input a torch.tensor of shape (b,nd), Output a boolean torch.tensor of shape (nd)
        """
        masks = [None] * len(self.filterparams)
        for i, param in enumerate(self.filterparams):
            masks[i] = torch.sum( (x - torch.tensor(param[:2], device=x.device))**2, dim=1) > param[2]**2
        for i in range(1,len(self.filterparams)):
            masks[i] = masks[i] & masks[i-1]
        return masks[-1]
    
    def sample_bd(self, N_bd):
        """
        Sample (approximately) N_bd points on the boundary
        """
        pts = []
        N_bd = (N_bd + 7) // 8
        for i in range(N_bd):
            pts.append([-self.xscale/2, -self.xscale/2 + self.xscale*i/N_bd])
            pts.append([-self.xscale/2 + self.xscale*i/N_bd, self.xscale/2])
            pts.append([self.xscale/2, self.xscale/2 - self.xscale*i/N_bd])
            pts.append([self.xscale/2 - self.xscale*i/N_bd, -self.xscale/2])
        for i in range(N_bd):
            theta = 2*np.pi*i/N_bd
            for param in self.filterparams:
                pts.append([param[0]+param[2]*np.cos(theta), param[1]+param[2]*np.sin(theta)])
        return pts
    
    def physics_loss(self, x, y, j0, j1, jj0, jj1):
        physics = jj0[:,0] + jj1[:,0]
        return losses.l2_loss(physics, 0)

    def bd_loss(self, x, y, j0, j1, jj0, jj1):
        is4sides = np.logical_or( np.isclose(np.abs(x[:,0].detach().cpu()), 0.5*self.xscale) , np.isclose(np.abs(x[:,1].detach().cpu()), 0.5*self.xscale) )
        is4sides = torch.tensor(is4sides, device=x.device)
        bd_true = torch.where(is4sides, 1., 0.)
        return losses.l2_loss(y[:,0],bd_true)

    def get_gradients(self, x, y):
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        jj0 = torch.autograd.grad(j0, x, torch.ones_like(j0), create_graph=True)[0][:,0:1]
        jj1 = torch.autograd.grad(j1, x, torch.ones_like(j1), create_graph=True)[0][:,1:2]
        return y, j0, j1, jj0, jj1

    def boundary_condition(self, x, y, j0, j1, jj0, jj1, sd):
        # do not use ansatz
        return y, j0, j1, jj0, jj1


class Poisson2DManyArea(_Problem):

    @property
    def name(self):
        return "Poisson2DManyArea"

    def __init__(self, bbox=[-10, 10, -10, 10], split=(5, 5)):
        self.bbox = bbox
        self.d = (2,1)
        self.mbbox, self.msplit = bbox, split
        self.a_cof = np.loadtxt("../ref/poisson_a_coef.dat")
        self.f_cof = np.loadtxt("../ref/poisson_f_coef.dat").reshape(5,5,2,2)
        self.load_ref_data("poisson_manyarea")
        # prepare interpn
        self.num_js = 4
    
    def prepare_a_f(self):
        # prepare a, f
        bbox, split = self.mbbox, self.msplit
        block_size = np.array([(bbox[1] - bbox[0] + 2e-5) / split[0], (bbox[3] - bbox[2] + 2e-5) / split[1]])
        def domain(x):
            reduced_x = (x - np.array(bbox[::2]) + 1e-5)
            dom = np.floor(reduced_x / block_size).astype("int32")
            return dom, reduced_x - dom * block_size

        def a(x):
            dom, res = domain(x)
            return self.a_cof[dom[0], dom[1]]

        self.a_vct = np.vectorize(a, signature="(2)->()")

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

        self.f_vct = np.vectorize(f, signature="(2)->()")
    
    @cache_x(maxsize=200)
    def get_coef(self, x):
        x_cpu = x.detach().cpu()
        if not hasattr(self, "a_vct"):
            self.prepare_a_f()
        return torch.tensor(self.a_vct(x_cpu),device=x.device), torch.tensor(self.f_vct(x_cpu),device=x.device)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy = j[:,0:1], j[:,1:2]
        jjx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jjy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        return y, jx, jy, jjx, jjy

    def physics_loss(self, x, y, jx, jy, jjx, jjy):
        a, f = self.get_coef(x)
        physics = a * (jjx + jjy) + f
        return losses.l2_loss(physics, 0)
    
    def sample_bd(self, N_bd):
        pts = []
        N_bd = (N_bd + 3) // 4
        for i in range(N_bd):
            pts.append([-10, -10+20*(i+0.5)/N_bd])
            pts.append([-10+20*(i+0.5)/N_bd, 10])
            pts.append([10, 10-20*(i+0.5)/N_bd])
            pts.append([10-20*(i+0.5)/N_bd, -10])
        return pts

    def bd_loss(self, x, y, jx, jy, jjx, jjy):
        isl = np.isclose(x[:,0].detach().cpu(), -10)
        isr = np.isclose(x[:,0].detach().cpu(), 10)
        isb = np.isclose(x[:,1].detach().cpu(), -10)
        ist = np.isclose(x[:,1].detach().cpu(), 10)
        isl, isr, isb, ist = (torch.tensor(i, device=x.device) for i in (isl, isr, isb, ist))
        normal_deri = torch.sum(torch.stack([torch.where(cond, j[:,0], 0.) for cond, j in [(isl,-jx), (isr,jx), (isb,-jy), (ist,jy)]]), dim=0)
        return losses.l2_loss(normal_deri+y[:,0], 0)


class Poisson3D(_Problem):
    @property
    def name(self):
        return "Poisson3D"
    
    def __init__(self):
        self.d = (3, 1)
        self.bbox=[0, 1, 0, 1, 0, 1]
        self.interface_z=0.5
        self.circs=[(0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2), (0.2, 0.8, 0.7, 0.1), (0.6, 0.2, 0.3, 0.1)]
        self.A_param=(20, 100)
        self.m_param=(1, 10, 5)
        self.k_param=(8, 10)
        self.mu_param=(1, 1)
        self.load_ref_data("poisson_3d")
        self.num_js=6
    
    def get_gradients(self, x, y):
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jz = j[:,0:1], j[:,1:2], j[:,2:3]
        jxx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jyy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        jzz = torch.autograd.grad(jz, x, torch.ones_like(jz), create_graph=True)[0][:,2:3]
        return y, jx, jy, jz, jxx, jyy, jzz

    def physics_loss(self, x, y, jx, jy, jz, jxx, jyy, jzz):
        def f(xyz):
            x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
            xlen2 = x**2 + y**2 + z**2
            part1 = torch.exp(torch.sin(self.m_param[0] * x) + torch.sin(self.m_param[1] * y) + torch.sin(self.m_param[2] * z)) * (xlen2 - 1) / (xlen2 + 1)
            part2 = torch.sin(self.m_param[0] * torch.pi * x) + torch.sin(self.m_param[1] * torch.pi * y) + torch.sin(self.m_param[2] * torch.pi * z)
            return self.A_param[0] * part1 + self.A_param[1] * part2
        mus = torch.where(x[:, 2] < self.interface_z, self.mu_param[0], self.mu_param[1]).unsqueeze(dim=-1)
        ks = torch.where(x[:, 2] < self.interface_z, self.k_param[0]**2, self.k_param[1]**2).unsqueeze(dim=-1)
        physics = -mus * (jxx + jyy + jzz) + ks * y - f(x)
        return losses.l2_loss(physics, 0)
    
    def mask_x(self, x):
        x_cpu = x.detach().cpu()
        masks_all = torch.ones_like(x_cpu[:,0], dtype=torch.bool)        
        for circ in self.circs:
            masks_all = masks_all & (torch.sum( (x_cpu - torch.tensor(circ[:3]))**2, dim=1) >= circ[3]**2)
        return masks_all.to(x.device)
    
    def sample_bd(self, N_bd):
        nside = int(np.sqrt(N_bd//6))
        def mgrid(x1, x2, nx, y1, y2, ny, d3idx, d3val):
            xl, yl = np.linspace(x1, x2, nx), np.linspace(y1, y2, ny)
            xmesh, ymesh = np.meshgrid(*(xl, yl), indexing='ij')
            zmesh = np.ones_like(xmesh, dtype=xmesh.dtype) * d3val
            meshes = [xmesh, ymesh]
            meshes.insert(d3idx, zmesh)
            return np.stack(meshes, axis=-1).reshape(-1,3)
        bc_z0 = mgrid(0, 1, nside, 0, 1, nside, 2, 0)
        bc_z1 = mgrid(0, 1, nside, 0, 1, nside, 2, 1)
        bc_y0 = mgrid(0, 1, nside, 0, 1, nside, 1, 0)
        bc_y1 = mgrid(0, 1, nside, 0, 1, nside, 1, 1)
        bc_x0 = mgrid(0, 1, nside, 0, 1, nside, 0, 0)
        bc_x1 = mgrid(0, 1, nside, 0, 1, nside, 0, 1)
        bd_pts = np.concatenate([bc_z0, bc_z1, bc_y0,bc_y1,bc_x0,bc_x1], axis=0)
        return bd_pts
    
    def bd_loss(self, x, y, jx, jy, jz, jxx, jyy, jzz):
        x_cpu = x.detach().cpu()
        isx, isy, isz = (torch.tensor(np.isclose(x_cpu[:,idim], 1) | np.isclose(x_cpu[:,idim], 0)) for idim in (0,1,2))
        xloss, yloss, zloss = (torch.where(isd.to(x.device), jd, 0.) for isd, jd in zip((isx, isy, isz),(jx, jy, jz)))
        return losses.l2_loss(torch.concat((xloss, yloss, zloss), dim=1), 0)



class PoissonND(_Problem):

    @property
    def name(self):
        return "PoissonND"
    
    def __init__(self, dim=5):
        self.bbox = [0, 1]*dim
        self.d = (dim,1)
        self.xdim = dim
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jjsum = torch.zeros_like(y)
        for idim in range(self.xdim):
            ji = j[:,idim:idim+1]
            jjsum += torch.autograd.grad(ji, x, torch.ones_like(ji), create_graph=True)[0][:,idim:idim+1]
        return y, jjsum
    
    def f(self, x):
        return torch.sin(torch.pi/2 * x).sum(axis=1).reshape(-1,1)
    
    def physics_loss(self, x, y, jjsum):
        physics = jjsum + (torch.pi**2)/4 * self.f(x)
        return losses.l2_loss(physics, 0)
    
    def exact_solution(self, x, batch_size):
        y_exact = self.f(x)
        jjsum_exact = -(torch.pi**2)/4 * self.f(x)
        return y_exact, jjsum_exact
    
    def sample_bd(self, N_bds):
        def hyperplane(keepdim):
            vardims = [np.linspace(0,1,8) for _ in range(self.xdim-1)]
            varmeshes = [item for item in np.meshgrid(*vardims)]
            keepmesh_0 = np.zeros_like(varmeshes[0],dtype=varmeshes[0].dtype)
            keepmesh_1 = np.ones_like(varmeshes[0],dtype=varmeshes[0].dtype)
            varmeshes.insert(keepdim, keepmesh_0)
            ret_0 = np.stack(varmeshes, axis=-1).reshape(-1,self.xdim)
            varmeshes[keepdim] = keepmesh_1
            ret_1 = np.stack(varmeshes, axis=-1).reshape(-1,self.xdim)
            return np.stack([ret_0, ret_1], axis=0).reshape(-1,self.xdim)
        def hyperplane_random(keepdim):
            ret = torch.rand((N_bds,self.xdim))
            ret[:N_bds//2,keepdim] = 0.
            ret[N_bds//2:,keepdim] = 1.
            return ret
        retlist = list()
        for idim in range(self.xdim):
            retlist.append(hyperplane_random(idim))
        return np.stack(retlist, axis=0).reshape(-1,self.xdim)
    
    def bd_loss(self, x, y, jjsum):
        return losses.l2_loss(self.f(x),y)