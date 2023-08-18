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

# Obsl
class CavityFlowPlus(_Problem):
    """
    Solves the 2x3D problem:
    Domain: [0,1] x [0,1]
    Unknowns: Velocity vector field {u(x,y), v(x,y)}, pressure p(x,y)
    Eqns: 
    (u \cdot nabla) u + nabla p = 1/Re Laplace u
    nabla \cdot u = 0
    Boundary conditions:
    u(x,1) = 1, v(x,1) = 0; u,v = 0 on other boundary sides
    Ansatz:
    None
    """
    @property
    def name(self):
        return "Cavityflow_BD"
    
    def __init__(self):
        # dimensionality of x and y
        # u, v, p = y[:,0], y[:,1], y[:,2]
        self.d = (2,3)
        self.re = 100
        u_ref= np.genfromtxt("../analytical_solutions/reference_u.csv", delimiter=',').flatten()
        self.u_ref = torch.tensor(u_ref)
        v_ref= np.genfromtxt("../analytical_solutions/reference_v.csv", delimiter=',').flatten()
        self.v_ref = torch.tensor(v_ref)
    
    def sample_bd(self, N_bd):
        """
        Sample (approximately) N_bd points on the boundary
        """
        pts = []
        N_bd = (N_bd + 3) // 4
        for i in range(N_bd):
            pts.append([0, i/N_bd])
            pts.append([i/N_bd, 1])
            pts.append([1, 1-i/N_bd])
            pts.append([1-i/N_bd, 0])
        return pts

    def bd_loss(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y):
        istop = np.isclose(x[:,1].detach().cpu(), 1)
        istop = torch.tensor(istop)
        bd_true_u = torch.where(istop, 1., 0.)
        bd_true_v = torch.where(istop, 0., 0.)
        return losses.l2_loss(y[:,:2],torch.stack((bd_true_u, bd_true_v), dim=1).to(y.device))

    def physics_loss(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y):
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        momentum_u = u*u_x + v*u_y + p_x - (1/self.re) * (u_xx + u_yy)
        momentum_v = u*v_x + v*v_y + p_y - (1/self.re) * (v_xx + v_yy)
        continuity = u_x + v_y
        physics = torch.concat((momentum_u, momentum_v, continuity), dim=1)
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        j_u = torch.autograd.grad(y[:,0], x, torch.ones_like(y[:,0]), create_graph=True)[0]
        u_x, u_y = j_u[:,0:1], j_u[:,1:2]
        j_v = torch.autograd.grad(y[:,1], x, torch.ones_like(y[:,1]), create_graph=True)[0]
        v_x, v_y = j_v[:,0:1], j_v[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][:,0:1]
        v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][:,1:2]
        j_p = torch.autograd.grad(y[:,2], x, torch.ones_like(y[:,2]), create_graph=True)[0]
        p_x, p_y = j_p[:,0:1], j_p[:,1:2]
        return y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y

    def boundary_condition(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y, sd):
        return y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y

    def exact_solution(self, x, batch_size):
        assert x.shape[0] == 10000
        ref_y = torch.stack((self.u_ref, self.v_ref, torch.zeros_like(self.u_ref)), dim=1).to(x.device)
        return (ref_y,) + (torch.zeros( (np.prod(batch_size),) + (1,), device=x.device ), )*10 # shallow copy, but yj_true is read only


class LidDrivenFlow(_Problem):
    @property
    def name(self):
        return "LidDrivenFlow"
    
    def __init__(self, param_a = 4):
        self.bbox = [0, 1, 0, 1]
        self.d = (2,3) # x, y; u, v, p
        self.re = 100
        self.param_a = param_a
        self.load_ref_data("lid_driven_a"+str(param_a))
        self.num_js = 10
    
    def param2str(self):
        return str(self.param_a)

    def sample_bd(self, N_bd):
        """
        Sample (approximately) N_bd points on the boundary
        """
        pts = []
        N_bd = (N_bd + 3) // 4
        for i in range(N_bd):
            pts.append([0, i/N_bd])
            pts.append([i/N_bd, 1])
            pts.append([1, 1-i/N_bd])
            pts.append([1-i/N_bd, 0])
        return pts

    def bd_loss(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y):
        istop = np.isclose(x[:,1].detach().cpu(), 1)
        istop = torch.tensor(istop)
        bd_true_u = torch.where(istop, self.param_a*x[:,0].cpu()*(1-x[:,0].cpu()), 0.) # u=a*x(1-x) at top
        bd_true_v = torch.where(istop, 0., 0.)
        return losses.l2_loss(y[:,:2],torch.stack((bd_true_u, bd_true_v), dim=1).to(y.device))

    def physics_loss(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y):
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        momentum_u = u*u_x + v*u_y + p_x - (1/self.re) * (u_xx + u_yy)
        momentum_v = u*v_x + v*v_y + p_y - (1/self.re) * (v_xx + v_yy)
        continuity = u_x + v_y
        physics = torch.concat((momentum_u, momentum_v, continuity), dim=1)
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        j_u = torch.autograd.grad(y[:,0], x, torch.ones_like(y[:,0]), create_graph=True)[0]
        u_x, u_y = j_u[:,0:1], j_u[:,1:2]
        j_v = torch.autograd.grad(y[:,1], x, torch.ones_like(y[:,1]), create_graph=True)[0]
        v_x, v_y = j_v[:,0:1], j_v[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][:,0:1]
        v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][:,1:2]
        j_p = torch.autograd.grad(y[:,2], x, torch.ones_like(y[:,2]), create_graph=True)[0]
        p_x, p_y = j_p[:,0:1], j_p[:,1:2]
        return y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y


class NS_Long(_Problem):
    @property
    def name(self):
        return "NS_Long"
    
    def __init__(self):
        self.bbox=[0, 2, 0, 1, 0, 5]
        # u, v, p = y[:,0], y[:,1], y[:,2]
        self.d = (3,3)
        self.nu = 1/100
        self.load_ref_data("ns_long", timepde=(0, 5))
        # downsample on t
        #m1 = np.array([True, False, False, False, False])
        #m = np.concatenate([m1,m1,m1,m1,m1,m1,m1,m1,m1,m1,np.array([True])])
        #self.ref_data = self.ref_data.reshape(-1,51,6)[:,m,:].reshape(-1,6)
        # debug
        #for i in range(5):
        #    print(self.ref_data[i])
        self.downsample_ref_data(6)
        self.num_js = 12
    
    def get_gradients(self, x, y):
        j_u = torch.autograd.grad(y[:,0], x, torch.ones_like(y[:,0]), create_graph=True)[0]
        u_x, u_y, u_t = j_u[:,0:1], j_u[:,1:2], j_u[:,2:3]
        j_v = torch.autograd.grad(y[:,1], x, torch.ones_like(y[:,1]), create_graph=True)[0]
        v_x, v_y, v_t = j_v[:,0:1], j_v[:,1:2], j_v[:,2:3]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][:,0:1]
        v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][:,1:2]
        j_p = torch.autograd.grad(y[:,2], x, torch.ones_like(y[:,2]), create_graph=True)[0]
        p_x, p_y = j_p[:,0:1], j_p[:,1:2]
        return y, u_x, u_y, u_t, u_xx, u_yy, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y
    
    def physics_loss(self, x, y, u_x, u_y, u_t, u_xx, u_yy, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y):
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        force = torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2]) * torch.sin(np.pi * x[:, 2:3])
        momentum_u = u_t + u*u_x + v*u_y + p_x - self.nu * (u_xx + u_yy) - force
        momentum_v = v_t + u*v_x + v*v_y + p_y - self.nu * (v_xx + v_yy) - force
        continuity = u_x + u_y
        return losses.l2_loss(torch.concat((momentum_u, momentum_v, continuity), dim=1), 0)
    
    def boundary_condition(self, x, y, u_x, u_y, u_t, u_xx, u_yy, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y, sd):
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        t0, jt0 = boundary_conditions.tanh_1(x[:,2:3], 0, sd)
        t1, t1y, t1yy = boundary_conditions.tanhtanh_2(x[:,1:2], 0, 1, sd)
        tr2, tr2x = boundary_conditions.tanh_1(x[:,0:1], 2, sd) # p = 0 at exit (x=2)
        tl2, tl2x, tl2xx = boundary_conditions.tanh_2(x[:,0:1], 0, sd) # v = 0 at entry (x=0)
        ux, vx = (t0*t1*z_x for z_x in (u_x, tl2*v_x+tl2x*v))
        uxx, vxx = (t0*t1*z_xx for z_xx in (u_xx, tl2*v_xx+2*tl2x*v_x+tl2xx*v))
        uy, vy = (t0*t1*z_y+t0*t1y*z for z,z_y in ((u, u_y), (tl2*v, tl2*v_y)))
        uyy, vyy = (t0*t1*z_yy+2*t0*t1y*z_y+t0*t1yy*z for z,z_y,z_yy in ((u, u_y, u_yy), (tl2*v, tl2*v_y, tl2*v_yy)))
        ut, vt = (t0*t1*z_t+jt0*t1*z for z,z_t in ((u, u_t), (tl2*v, tl2*v_t)))
        px, py = t0*tr2*p_x+t0*tr2x*p, t0*tr2*p_y
        return torch.concat((u*t0*t1, v*t0*t1*tl2, p*t0*tr2), dim=1), ux,uy,ut,uxx,uyy, vx,vy,vt,vxx,vyy, px,py
    
    def sample_bd(self, N_bd):
        bd_pts = torch.tensor([0., 0., 0.])+torch.tensor([0., 1., 5.])*torch.rand((N_bd,3))
        return bd_pts.numpy()
    
    def bd_loss(self, x, y, u_x, u_y, u_t, u_xx, u_yy, v_x, v_y, v_t, v_xx, v_yy, p_x, p_y):
        u_true = torch.where(y[:,1:2] <= 0.5, torch.sin(np.pi*x[:,1:2])*(torch.sin(np.pi*x[:,2:3])+torch.sin(3*np.pi*x[:,2:3])+torch.sin(5*np.pi*x[:,2:3])), 0.)
        return losses.l2_loss(u_true, y[:,0:1])


# Obsl
class NS_FourCircles(_Problem):
    @property
    def name(self):
        return "NS_FourCircles"
    
    def __init__(self):
        self.bbox=[0, 4, 0, 2]
        # u, v, p = y[:,0], y[:,1], y[:,2]
        self.d = (2,3)
        self.nu = 1/100
        self.load_ref_data("ns_4_obstacle")
        self.circs = [(1, 0.5, 0.2), (2, 0.5, 0.3), (2.7, 1.6, 0.2)]
        self.num_js = 10
    
    def get_gradients(self, x, y):
        j_u = torch.autograd.grad(y[:,0], x, torch.ones_like(y[:,0]), create_graph=True)[0]
        u_x, u_y = j_u[:,0:1], j_u[:,1:2]
        j_v = torch.autograd.grad(y[:,1], x, torch.ones_like(y[:,1]), create_graph=True)[0]
        v_x, v_y = j_v[:,0:1], j_v[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][:,0:1]
        v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][:,1:2]
        j_p = torch.autograd.grad(y[:,2], x, torch.ones_like(y[:,2]), create_graph=True)[0]
        p_x, p_y = j_p[:,0:1], j_p[:,1:2]
        return y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y
    
    def physics_loss(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y):
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        momentum_u = u*u_x + v*u_y + p_x - self.nu * (u_xx + u_yy)
        momentum_v = u*v_x + v*v_y + p_y - self.nu * (v_xx + v_yy)
        continuity = u_x + v_y
        physics = torch.concat((momentum_u, momentum_v, continuity), dim=1)
        return losses.l2_loss(physics, 0)
    
    def boundary_condition(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y, sd):
        """ Ansatz:
        u = u_func(y) + tanh(x)*u
        v = tanh(x)*v
        p = tanh(x-4)*p
        """
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        tl0, jtl0, jjtl0 = boundary_conditions.tanh_2(x[:,0:1], 0, sd)
        tr0, jtr0 = boundary_conditions.tanh_1(x[:,0:1], 4, sd)
        uf, juf, jjuf = x[:,1:2]*(1-x[:,1:2])/4, (1-2*x[:,1:2])/4, -0.5
        ux, uy = tl0*u_x+jtl0*u, juf+tl0*u_y
        uxx, uyy = tl0*u_xx+2*jtl0*u_x+jjtl0*u, jjuf+tl0*u_yy
        vx, vy = tl0*v_x+jtl0*v, tl0*v_y
        vxx, vyy = tl0*v_xx+2*jtl0*v_x+jjtl0*v, tl0*v_yy
        px, py = tr0*p_x+jtr0*p, tr0*p_y
        return torch.concat((uf+tl0*u,tl0*v,tr0*p), dim=1), ux, uy, uxx, uyy, vx, vy, vxx, vyy, px, py
    
    def mask_x(self, x):
        x_cpu = x.detach().cpu()
        masks_all = torch.ones_like(x_cpu[:,0], dtype=torch.bool)        
        for circ in self.circs:
            masks_all = masks_all & (torch.sum( (x_cpu[:,:2] - torch.tensor([circ[0], circ[1]]))**2, dim=1) > circ[2]**2)
        masks_all = masks_all & ((x_cpu[:,0]>=2) | (x_cpu[:,1]<=1))
        masks_all = masks_all & ((x_cpu[:,0]<=2.8) | (x_cpu[:,0]>=3.6) | (x_cpu[:,1]<=0.8) | (x_cpu[:,1]>=1.1))
        return masks_all.to(x.device)
    
    def sample_bd(self, N_bd):
        pts = []
        N_bd = N_bd // 7
        for circ in self.circs:
            for i in range(N_bd):
                theta = 2*np.pi*i/N_bd
                pts.append([circ[0]+circ[2]*np.cos(theta), circ[1]+circ[2]*np.sin(theta)])
        N_rect = N_bd//4
        for i in range(N_rect):
            pts.append([2.8, 0.8+0.3*i/N_rect])
            pts.append([2.8+0.8*i/N_rect, 1.1])
            pts.append([3.6, 1.1-0.3*i/N_rect])
            pts.append([3.6-0.8*8/N_rect, 0.8])
        for i in range(N_bd):
            pts.append([4*i/N_bd, 0])
            pts.append([2*i/N_bd, 1])
            pts.append([2+2*i/N_bd, 2])
        return pts
    
    def bd_loss(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y):
        return losses.l2_loss(y[:,:2], 0)


class NS_NoObstacle(_Problem):
    @property
    def name(self):
        return "NS_NoObstacle"
    
    def __init__(self):
        self.bbox=[0, 4, 0, 2]
        # u, v, p = y[:,0], y[:,1], y[:,2]
        self.d = (2,3)
        self.nu = 1/100
        self.load_ref_data("ns_0_obstacle")
        self.num_js = 10
    
    def get_gradients(self, x, y):
        j_u = torch.autograd.grad(y[:,0], x, torch.ones_like(y[:,0]), create_graph=True)[0]
        u_x, u_y = j_u[:,0:1], j_u[:,1:2]
        j_v = torch.autograd.grad(y[:,1], x, torch.ones_like(y[:,1]), create_graph=True)[0]
        v_x, v_y = j_v[:,0:1], j_v[:,1:2]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, x, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0][:,0:1]
        v_yy = torch.autograd.grad(v_y, x, torch.ones_like(v_y), create_graph=True)[0][:,1:2]
        j_p = torch.autograd.grad(y[:,2], x, torch.ones_like(y[:,2]), create_graph=True)[0]
        p_x, p_y = j_p[:,0:1], j_p[:,1:2]
        return y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y
    
    def physics_loss(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y):
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        momentum_u = u*u_x + v*u_y + p_x - self.nu * (u_xx + u_yy)
        momentum_v = u*v_x + v*v_y + p_y - self.nu * (v_xx + v_yy)
        continuity = u_x + v_y
        physics = torch.concat((momentum_u, momentum_v, continuity), dim=1)
        return losses.l2_loss(physics, 0)
    
    def boundary_condition(self, x, y, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y, sd):
        """ Ansatz:
        u = u_func(y) + tanh(y)tanh(2-y)tanh(x)*u
        v = tanh(y)tanh(2-y)tanh(x)*v
        p = tanh(x-4)*p
        """
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        tl0, jtl0, jjtl0 = boundary_conditions.tanh_2(x[:,0:1], 0, sd)
        ty0, jty0, jjty0 = boundary_conditions.tanhtanh_2(x[:,1:2],0,2,sd)
        tr0, jtr0 = boundary_conditions.tanh_1(x[:,0:1], 4, sd)
        uf, juf, jjuf = 4*x[:,1:2]*(1-x[:,1:2]), 4*(1-2*x[:,1:2]), -8
        ux, uy = ty0*(tl0*u_x+jtl0*u), juf+tl0*(ty0*u_y+jty0*u)
        uxx, uyy = ty0*(tl0*u_xx+2*jtl0*u_x+jjtl0*u), jjuf+tl0*(ty0*u_yy+2*jty0*u_y+jjty0*u)
        vx, vy = ty0*(tl0*v_x+jtl0*v), tl0*(ty0*v_y+jty0*v)
        vxx, vyy = ty0*(tl0*v_xx+2*jtl0*v_x+jjtl0*v), tl0*(ty0*v_yy+2*jty0*v_y+jjty0*v)
        px, py = tr0*p_x+jtr0*p, tr0*p_y
        return torch.concat((uf+ty0*tl0*u,ty0*tl0*v,tr0*p), dim=1), ux, uy, uxx, uyy, vx, vy, vxx, vyy, px, py