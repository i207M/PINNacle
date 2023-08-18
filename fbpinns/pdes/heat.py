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


class HeatMultiscale(_Problem):
    """
    Solves the 2d heat multiscale problem
    Using the ansatz u(x,y,t) = u(x,y,0) + NN(x,y,t)*tanh(wt)*tanh(x)tanh(x-1)tanh(y)tanh(y-1)
    """
    @property
    def name(self):
        return "HeatMultiscale"
    
    def __init__(self):
        self.bbox =[0, 1, 0, 1, 0, 5]
        self.d = (3,1) # input dim order: x, y, t
        self.load_ref_data("heat_multiscale_lesspoints", timepde=(0, 5))
        self.nx = 20
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jt = j[:,0:1], j[:,1:2], j[:,2:3]
        jjx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jjy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        return y, jt, jx, jy, jjx, jjy

    def physics_loss(self, x, y, jt, jx, jy, jjx, jjy):
        physics = jt[:,0] - 1/(500*np.pi)**2 * jjx[:,0] - 1/(np.pi)**2 * jjy[:,0]
        return losses.l2_loss(physics, 0)

    def boundary_condition(self, x, y, jt, jx, jy, jjx, jjy, sd):
        # TBD
        t0, jt0 = boundary_conditions.tanh_1(x[:,2:3], 0, sd)
        tx, jtx, jjtx = boundary_conditions.tanhtanh_2(x[:,0:1], 0, 1, sd)
        ty, jty, jjty = boundary_conditions.tanhtanh_2(x[:,1:2], 0, 1, sd)
        u_new = torch.sin(self.nx*np.pi*x[:,0:1]) * torch.sin(np.pi*x[:,1:2]) + y * t0 * tx * ty
        u_new_t = (jt * t0 + y * jt0) * tx * ty
        u_new_x = (self.nx*np.pi) * torch.cos(self.nx*np.pi*x[:,0:1]) * torch.sin(np.pi*x[:,1:2]) + (y*jtx + jx*tx) * t0 * ty
        u_new_xx = -(self.nx*np.pi)**2 * torch.sin(self.nx*np.pi*x[:,0:1]) * torch.sin(np.pi*x[:,1:2]) + (y*jjtx + 2*jx*jtx + jjx*tx) * t0 * ty
        u_new_y = (np.pi) * torch.cos(np.pi*x[:,1:2]) * torch.sin(self.nx*np.pi*x[:,0:1]) + (y*jty + jy*ty) * t0 * tx
        u_new_yy = -(np.pi)**2 * torch.sin(np.pi*x[:,1:2]) * torch.sin(self.nx*np.pi*x[:,0:1]) + (y*jjty + 2*jy*jty + jjy*ty) * t0 * tx
        return u_new, u_new_t, u_new_x, u_new_y, u_new_xx, u_new_yy
    
    def exact_solution(self, x, batch_size):
        eps = 1e-8
        x_lims = [ (x[:,in_dim].cpu().min(), x[:,in_dim].cpu().max()) for in_dim in range(self.d[0])]
        x_mesh = [np.linspace(lim[0]+eps, lim[1]-eps, b) for lim,b in zip(x_lims, batch_size)]
        grid_x_tup = np.meshgrid(*x_mesh, indexing="ij")
        print(np.min(self.ref_x,axis=0), np.max(self.ref_x,axis=0))
        m1 = np.array([True, False, False, False, False])
        m = np.concatenate([m1,m1,m1,m1,m1,np.array([True])])
        temp = self.ref_x.reshape(-1, 26, 3)[:,m,:].reshape(-1,3)
        print(np.min(temp,axis=0), np.max(temp,axis=0))
        cache_str = "interpolate_cache/"+self.name+"_".join([str(n) for n in batch_size])+".pkl"
        try:
            grid_interp = pickle.load(open(cache_str,'rb'))
        except FileNotFoundError:
            with Timer("interpolate"):
                grid_interp = griddata(self.ref_x.reshape(-1,26,3)[:,m,:].reshape(-1,3), self.ref_y.reshape(-1,26,1)[:,m,:].reshape(-1,1), tuple(grid_x_tup) ) # should change code if ref_y is multidimentional
            pickle.dump(grid_interp, open(cache_str,'wb'))
        # use torch.ones to prevent divide by 0 (-> nan) in L2RE calculation
        return (torch.tensor(grid_interp.reshape(-1, 1), device=x.device),) + (torch.ones((np.prod(batch_size),1), device=x.device),)*5


class HeatComplex(_Problem):
    @property
    def name(self):
        return "HeatComplex"

    def __init__(self):
        self.bbox = [-8, 8, -12, 12, 0, 3]
        self.d = (3, 1) # x, y, t
        self.load_ref_data("heat_complex", timepde=(0, 3))
        #m1 = np.array([True, False, True, False, False])
        #m = np.concatenate([m1,m1,m1,m1,m1,m1,np.array([True])])
        # downsample on dim of t
        self.downsample_ref_data(3)
        self.num_js = 5
        self.big_centers = [(-4, -3), (4, -3), (-4, 3), (4, 3), (-4, -9), (4, -9), (-4, 9), (4, 9), (0, 0), (0, 6), (0, -6)]
        self.small_centers = [(-3.2, -6), (-3.2, 6), (3.2, -6), (3.2, 6), (-3.2, 0), (3.2, 0)]
        self.big_r = 1.0
        self.small_r = 0.4

    @cache_x()
    def mask_x(self, x):
        """
        Input a torch.tensor of shape (b,nd), Output a boolean torch.tensor of shape (nd)
        """
        x_cpu = x.detach().cpu()
        masks_all = torch.ones_like(x_cpu[:,0], dtype=torch.bool)
        for centers, r in [(self.big_centers, self.big_r), (self.small_centers, self.small_r)]:
            masks = [None] * len(centers)
            for i, center in enumerate(centers):
                masks[i] = torch.sum( (x_cpu[:,:2] - torch.tensor(center))**2, dim=1) > r**2
            for i in range(1,len(centers)):
                masks[i] = masks[i] & masks[i-1]
            masks_all = masks_all & masks[-1]
        return masks_all.to(x.device)
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jt = j[:,0:1], j[:,1:2], j[:,2:3]
        jjx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jjy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        return y, jt, jx, jy, jjx, jjy
    
    def physics_loss(self, x, y, jt, jx, jy, jjx, jjy):
        physics = jt - jjx - jjy
        return losses.l2_loss(physics, 0)

    def boundary_condition(self, x, y, jt, jx, jy, jjx, jjy, sd):
        # u(x,y,t) = 0 + NN(x,y,t) tanh(t)
        t0, jt0 = boundary_conditions.tanh_1(x[:,2:3], 0, sd)
        y_new = y * t0
        jt_new = y * jt0 + jt * t0
        return y_new, jt_new, t0*jx, t0*jy, t0*jjx, t0*jjy

    def sample_bd(self, N_bd):
        pts = []
        h_div = 11
        N_bd = (N_bd + h_div - 1) // h_div
        def add_h(pts):
            ret = []
            for ih in range(h_div):
                h = 3 * ih / h_div
                for pt in pts:
                    ret.append(pt+[h])
            return ret
        n_big_circ = len(self.big_centers)
        n_small_circ = len(self.small_centers)
        n_circ = n_big_circ + n_small_circ
        N_bd_circ = (N_bd + n_circ - 1) // n_circ
        for i in range(N_bd_circ):
            theta = 2*np.pi*i/N_bd_circ
            for center in self.big_centers:
                pts.append([center[0]+self.big_r*np.cos(theta), center[1]+self.big_r*np.sin(theta)])
            for center in self.small_centers:
                pts.append([center[0]+self.small_r*np.cos(theta), center[1]+self.small_r*np.sin(theta)])
        N_bd_edge = (N_bd + 3) // 4
        for i in range(N_bd_edge):
            pts.append([-8, -12+24*(i+0.5)/N_bd_edge])
            pts.append([-8+16*(i+0.5)/N_bd_edge, 12])
            pts.append([8, 12-24*(i+0.5)/N_bd_edge])
            pts.append([8-16*(i+0.5)/N_bd_edge, -12])
        return add_h(pts)
    
    def bd_loss(self, x, y, jt, jx, jy, jjx, jjy):
        def nv(x,c,r):
            return (torch.tensor([c[0],c[1]],dtype=torch.float32,device=x.device) - x[:,:2]) / r # normal vector of holes point inwards
        def dot2d(nv,jx,jy):
            return nv[:,0]*jx[:,0] + nv[:,1]*jy[:,0]
        isl = np.isclose(x[:,0].detach().cpu(), -8)
        isr = np.isclose(x[:,0].detach().cpu(), 8)
        isb = np.isclose(x[:,1].detach().cpu(), -12)
        ist = np.isclose(x[:,1].detach().cpu(), 12)
        isl, isr, isb, ist = (torch.tensor(i, device=x.device) for i in (isl, isr, isb, ist))
        d_edge = torch.sum(torch.stack([torch.where(cond, j[:,0], 0.) for cond, j in [(isl,0.1-y+jx), (isr,0.1-y-jx), (isb,0.1-y+jy), (ist,0.1-y-jy)]]), dim=0)
        is_smallcs = [np.isclose((x[:,0].detach().cpu()-center[0])**2 + (x[:,1].detach().cpu()-center[1])**2 , self.small_r**2) for center in self.small_centers]
        is_smallcs_ = [torch.tensor(i, device=x.device) for i in is_smallcs]
        d_small = torch.sum(torch.stack([torch.where(cond, 1-y[:,0]-dot2d(nv(x,c,self.small_r),jx,jy), 0.) for cond, c in zip(is_smallcs_, self.small_centers)]), dim=0)
        is_bigcs = [np.isclose((x[:,0].detach().cpu()-center[0])**2 + (x[:,1].detach().cpu()-center[1])**2 , self.big_r**2) for center in self.big_centers]
        is_bigcs_ = [torch.tensor(i, device=x.device) for i in is_bigcs]
        d_big = torch.sum(torch.stack([torch.where(cond, 5-y[:,0]-dot2d(nv(x,c,self.big_r),jx,jy), 0.) for cond, c in zip(is_bigcs_, self.big_centers)]), dim=0) # 1-y-dot2d leads to wrong tensorshape and wrong results
        d_all = d_edge + d_small + d_big
        return losses.l2_loss(d_all, 0)


class HeatLongTime(_Problem):
    @property
    def name(self):
        return "HeatLongTime"

    def __init__(self):
        self.bbox = [0, 1, 0, 1, 0, 100]
        self.d = (3, 1) #x, y, t
        self.load_ref_data("heat_longtime", timepde=(0, 100))
        self.num_js = 3
        self.k = 1
        self.m1, self.n1 = np.pi, np.pi
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jt = j[:,0:1], j[:,1:2], j[:,2:3]
        jjx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jjy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        return y, jt, jjx, jjy
    
    def physics_loss(self, x, y, jt, jjx, jjy):
        physics = jt - 0.001*(jjx+jjy) - 5*torch.sin(self.k * torch.square(y)) * (1+2*torch.sin(x[:,2:3]*np.pi/4)) * torch.sin(self.m1*x[:,0:1]) * torch.sin(self.n1*x[:,1:2])
        return losses.l2_loss(physics, 0)

    def boundary_condition(self, x, y, jt, jjx, jjy, sd):
        t0, jt0 = boundary_conditions.tanh_1(x[:,2:3], 0, sd)
        u0 = torch.sin(4*np.pi*x[:,0:1])*torch.sin(3*np.pi*x[:,1:2])
        y_new = y*t0 + u0
        yt_new = jt*t0 + y*jt0
        jjx_new = jjx*t0 - 16*np.pi**2*u0
        jjy_new = jjy*t0 - 9*np.pi**2*u0
        return y_new, yt_new, jjx_new, jjy_new
    
    def sample_bd(self, N_bd):
        nside = int(np.sqrt(N_bd//4))
        def mgrid(x1, x2, nx, y1, y2, ny, d3idx, d3val):
            xl, yl = np.linspace(x1, x2, nx), np.linspace(y1, y2, ny)
            xmesh, ymesh = np.meshgrid(*(xl, yl), indexing='ij')
            zmesh = np.ones_like(xmesh, dtype=xmesh.dtype) * d3val
            meshes = [xmesh, ymesh]
            meshes.insert(d3idx, zmesh)
            return np.stack(meshes, axis=-1).reshape(-1,3)
        bc_y0 = mgrid(0, 1, nside, 0, 100, nside, 1, 0)
        bc_y1 = mgrid(0, 1, nside, 0, 100, nside, 1, 1)
        bc_x0 = mgrid(0, 1, nside, 0, 100, nside, 0, 0)
        bc_x1 = mgrid(0, 1, nside, 0, 100, nside, 0, 1)
        bd_pts = np.concatenate([bc_y0,bc_y1,bc_x0,bc_x1], axis=0)
        return bd_pts
    
    def bd_loss(self, x, y, jt, jxx, jyy):
        return losses.l2_loss(y, 0)

class HeatDarcy(_Problem):
    @property
    def name(self):
        return "HeatDarcy"
    
    def __init__(self):
        self.bbox=[0, 1, 0, 1, 0, 5]
        self.d = (3, 1)
        self.load_ref_data("heat_darcy", timepde=(0, 5))
        self.A = 10
        self.m1, self.m2, self.m3 = 1,5,1
        self.heat_2d_coef = np.loadtxt("../ref/heat_2d_coef_256.dat")
        self.num_js = 5
        # prepare for interpn
        grid_nums = 256
        xs = self.heat_2d_coef[:256,0]
        ys = self.heat_2d_coef[::256,1]
        self.coef_in = (xs, ys)
        self.coef_values = self.heat_2d_coef[:,2].reshape(grid_nums, grid_nums)
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jt = j[:,0:1], j[:,1:2], j[:,2:3]
        jjx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jjy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        return y, jt, jx, jy, jjx, jjy
    
    def physics_loss(self, x, y, jt, jx, jy, jjx, jjy):
        ref_pts = tuple(xs.astype(np.float64) for xs in self.coef_in)
        intp_result = scipy.interpolate.interpn(ref_pts, self.coef_values.astype(np.float64), x[:,0:2].detach().cpu().numpy().astype(np.float64))
        coef_x = torch.tensor(intp_result.astype(np.float32), device=x.device)
        def f(x):
            return self.A * torch.sin(self.m1*np.pi*x[:,0]) * torch.sin(self.m2*np.pi*x[:,1]) * torch.sin(self.m3*np.pi*x[:,2])
        #print(coef_x.shape)
        #print(f(x).shape)
        physics = jt[:,0] - coef_x * (jjx[:,0] + jjy[:,0]) - f(x)
        #print(physics.shape)
        #print("done check")
        return losses.l2_loss(physics, 0)
    
    def boundary_condition(self, x, y, jt, jx, jy, jjx, jjy, sd):
        t0, jt0 = boundary_conditions.tanh_1(x[:,2:3], 0, sd)
        tx, jtx, jjtx = boundary_conditions.tanhtanh_2(x[:,0:1], 0, 1, sd)
        ty, jty, jjty = boundary_conditions.tanhtanh_2(x[:,1:2], 0, 1, sd)
        u_new = y*t0*tx*ty
        u_new_t = tx*ty*(jt*t0+y*jt0)
        u_new_x = t0*ty*(jx*tx+y*jtx)
        u_new_y = t0*tx*(jy*ty+y*jty)
        u_new_xx = t0*ty*(jjx*tx+2*jx*jtx+y*jjtx)
        u_new_yy = t0*tx*(jjy*ty+2*jy*jty+y*jjty)
        return u_new, u_new_t, u_new_x, u_new_y, u_new_xx, u_new_yy


class HeatND(_Problem):
    @property
    def name(self):
        return "HeatND"

    def __init__(self, spacedim=5):
        self.bbox=[0, 1]*spacedim + [0, 1]
        self.d = (spacedim+1, 1)

    def mask_x(self, x):
        rsquare = torch.square(x[:,:-1]).sum(axis=1)
        return rsquare < 1

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jjsum = torch.zeros_like(y)
        for idim in range(self.d[0]-1):
            ji = j[:,idim:idim+1]
            jjsum += torch.autograd.grad(ji, x, torch.ones_like(ji), create_graph=True)[0][:,idim:idim+1]
        return y, j, jjsum

    def physics_loss(self, x, y, j, jjsum):
        jt = j[:,-1:]
        rsquare = torch.square(x[:,:-1]).sum(axis=1).reshape(-1,1)
        f = -(1/(self.d[0]-1))*rsquare*torch.exp(0.5*rsquare+x[:,-1:])
        return losses.l2_loss(jt, (1/(self.d[0]-1))*jjsum + f)
    
    def exact_solution_(self, x):
        space = x[:,:-1]
        time = x[:,-1:]
        return torch.exp(torch.square(space).sum(axis=1).reshape(-1,1)+time)
    
    def exact_solution(self, x, batch_size):
        return self.exact_solution_(x), torch.ones((np.prod(batch_size),self.d[0]), device=x.device), \
            torch.ones((np.prod(batch_size),1), device=x.device)

    def sample_bd(self, N_bd):
        def unit_random_points_sphere(n, dim, random="pseudo"):
            X = np.random.normal(size=(n, dim)).astype(np.float32)
            row_norms = np.linalg.norm(X,ord=2,axis=1,keepdims=True)
            return X / row_norms
        pts = np.zeros((N_bd,self.d[0]),dtype=np.float32)
        pts[:,:self.d[0]-1] = unit_random_points_sphere(N_bd,self.d[0]-1)
        pts[:,self.d[0]-1] = np.random.rand(N_bd)
        return pts
    
    def bd_loss(self, x, y, j, jjsum):
        dotprod = torch.sum(x[:,:-1]*j[:,:-1],dim=-1,keepdim=True)
        return losses.l2_loss(dotprod, self.exact_solution_(x))

class HeatMultiscaleExact(_Problem):
    """
    Solves the 2d heat multiscale problem
    Using the ansatz u(x,y,t) = u(x,y,0) + NN(x,y,t)*tanh(wt)*tanh(x)tanh(x-1)tanh(y)tanh(y-1)
    Use the exact solution
    """
    @property
    def name(self):
        return "HeatMultiscaleExact"
    
    def __init__(self):
        self.bbox =[0, 1, 0, 1, 0, 5]
        self.d = (3,1) # input dim order: x, y, t
        self.nx = 20
        self.ny = 1
        self.c1 = 500*np.pi
        self.c2 = np.pi
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jy, jt = j[:,0:1], j[:,1:2], j[:,2:3]
        jjx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jjy = torch.autograd.grad(jy, x, torch.ones_like(jy), create_graph=True)[0][:,1:2]
        return y, jt, jx, jy, jjx, jjy

    def physics_loss(self, x, y, jt, jx, jy, jjx, jjy):
        physics = jt[:,0] - 1/(self.c1)**2 * jjx[:,0] - 1/(self.c2)**2 * jjy[:,0]
        return losses.l2_loss(physics, 0)

    def boundary_condition(self, x, y, jt, jx, jy, jjx, jjy, sd):
        t0, jt0 = boundary_conditions.tanh_1(x[:,2:3], 0, sd)
        tx, jtx, jjtx = boundary_conditions.tanhtanh_2(x[:,0:1], 0, 1, sd)
        ty, jty, jjty = boundary_conditions.tanhtanh_2(x[:,1:2], 0, 1, sd)
        u_new = torch.sin(self.nx*np.pi*x[:,0:1]) * torch.sin(self.ny*np.pi*x[:,1:2]) + y * t0 * tx * ty
        u_new_t = (jt * t0 + y * jt0) * tx * ty
        u_new_x = (self.nx*np.pi) * torch.cos(self.nx*np.pi*x[:,0:1]) * torch.sin(self.ny*np.pi*x[:,1:2]) + (y*jtx + jx*tx) * t0 * ty
        u_new_xx = -(self.nx*np.pi)**2 * torch.sin(self.nx*np.pi*x[:,0:1]) * torch.sin(self.ny*np.pi*x[:,1:2]) + (y*jjtx + 2*jx*jtx + jjx*tx) * t0 * ty
        u_new_y = (self.ny*np.pi) * torch.cos(self.ny*np.pi*x[:,1:2]) * torch.sin(self.nx*np.pi*x[:,0:1]) + (y*jty + jy*ty) * t0 * tx
        u_new_yy = -(self.ny*np.pi)**2 * torch.sin(self.ny*np.pi*x[:,1:2]) * torch.sin(self.nx*np.pi*x[:,0:1]) + (y*jjty + 2*jy*jty + jjy*ty) * t0 * tx
        return u_new, u_new_t, u_new_x, u_new_y, u_new_xx, u_new_yy
    
    def exact_solution(self, x, batch_size):
        decay = (self.nx*np.pi/self.c1)**2 + (self.ny*np.pi/self.c2)**2
        y_exact = torch.sin(self.nx*np.pi*x[:,0:1])*torch.sin(self.ny*np.pi*x[:,1:2])*torch.exp(-decay*x[:,2:3])
        y_exact.to(x.device)
        return (y_exact,) + (torch.ones((np.prod(batch_size),1), device=x.device),)*5