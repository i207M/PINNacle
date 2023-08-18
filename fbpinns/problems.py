#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:11:31 2021

@author: bmoseley
"""

# This module defines a set of PDE problems to solve
# Each problem is defined by a problem class, which must inherit from the _Problem base class
# Each problem class must define the NotImplemented methods which compute the PINN physics loss, 
# the gradients required to evaluate the PINN physics loss, the hard boundary conditions applied to
# the ansatz and the exact solution, if it exists.

# Problem classes are used by constants.py when defining FBPINN / PINN problems (and subsequently main.py)

import pickle
import numpy as np
from scipy.interpolate import griddata
import scipy
import torch

import boundary_conditions
import losses

import sys
sys.path.insert(0, './shared_modules')
from helper import Timer, cache_x



class _Problem:
    "Base problem class to be inherited by different problem classes"
    
    @property
    def name(self):
        "Defines a name string (only used for labelling automated training runs)"
        raise NotImplementedError
    
    def __init__(self):
        raise NotImplementedError
    
    def physics_loss(self, x, *yj):
        "Defines the PINN physics loss to train the NN"
        raise NotImplementedError
        
    def get_gradients(self, x, y):
        "Returns the gradients yj required for this problem"
        
    def boundary_condition(self, x, *yj_and_sd):
        "Defines the hard boundary condition to be applied to the NN ansatz"
        # Default: does nothing
        return yj_and_sd[:-1]
    
    def exact_solution(self, x, batch_size):
        "Defines exact solution if it exists, Default: use ref solution to interpolate"
        if hasattr(self, "ref_in_coords"): # reference solution is on regular grid
            vals = list()
            ref_pts = tuple(xs.astype(np.float64) for xs in self.ref_in_coords)
            for ref_val in self.ref_values:
                intp_result = scipy.interpolate.interpn(ref_pts, ref_val.astype(np.float64), x.cpu().numpy().astype(np.float64))
                vals.append( torch.tensor(intp_result.astype(np.float32), device=x.device) )
            vals = torch.stack(vals, dim=-1) #(-1, out_dims)
        else:
            param_str = self.param2str() if hasattr(self, "param2str") else ""
            cache_str = "interpolate_cache/"+self.name+"_"+param_str+"_"+"_".join([str(n) for n in batch_size])+".pkl"
            try:
                vals = pickle.load(open(cache_str,'rb'))
            except FileNotFoundError:
                with Timer("interpolate"):
                    vals = list()
                    for i_od in range(self.d[1]):
                        intp_result = scipy.interpolate.griddata(self.ref_x, self.ref_y[:,i_od], x.cpu().numpy(), fill_value=0)
                        vals.append( torch.tensor(intp_result.astype(np.float32), device=x.device) )
                    vals = torch.stack(vals, dim=-1)
                pickle.dump(vals, open(cache_str,'wb'))
        return (vals.to(x.device),) + (torch.ones((np.prod(batch_size),1), device=x.device),)*self.num_js

    def load_ref_data(self, name, timepde = None): # if pde is a timepde, then provide timepde=(t_start, t_end)
        datapath = "../ref/" + name + ".dat"
        f = open(datapath,'r',encoding="utf-8")
        self.ref_data = np.loadtxt(f, comments="%").astype(np.float32)
    
        if timepde != None: # transform ref_data
            time_starts, time_ends = timepde
            data = self.ref_data
            num_tsample = (data.shape[1] - (self.d[0] - 1) ) // self.d[1]
            assert num_tsample * self.d[1] == data.shape[1] - (self.d[0] - 1)
            t = np.linspace(time_starts, time_ends, num_tsample)
            t, x0 = np.meshgrid(t, data[:, 0]) # add the first input dimension that is not time
            list_x = [x0.reshape(-1)] # x0.reshape(-1) gives [e1,e1,...,e1, e2,e2,...,e2, ...] each element repeats num_tsample times (adjacent)
            for i in range(1, self.d[0]-1): # add other input dimensions that is not time
                list_x.append(np.stack([data[:,i] for _ in range(num_tsample)]).T.reshape(-1)) # each element repeats num_tsample times (adjacent)
            list_x.append(t.reshape(-1)) # t is the last input dimension. (Other) input dimension order should be in accordance with .dat file
            for i in range(self.d[1]):
                list_x.append(data[:, self.d[0]-1+i::self.d[1]].reshape(-1))
            self.ref_data = np.stack(list_x).T.astype(np.float32)
        
        self.ref_x = self.ref_data[:, :self.d[0]]
        self.ref_y = self.ref_data[:, self.d[0]:]
    
    def downsample_ref_data(self, factor):
        ndat = self.ref_data.shape[0]
        ia = np.random.choice(np.arange(ndat),ndat//factor)
        self.ref_data = self.ref_data[ia,:]
        self.ref_x = self.ref_x[ia,:]
        self.ref_y = self.ref_y[ia,:]


# 1D problems

class Cos1D_1(_Problem):
    """Solves the 1D ODE:
        du
        -- = cos(wx)
        dx
        
        Boundary conditions:
        u (0) = A
    """
    
    @property
    def name(self):
        return "Cos1D_1_w%s"%(self.w)
    
    def __init__(self, w, A=0):
        
        # input params
        self.w = w
        
        # boundary params
        self.A = A
        
        # dimensionality of x and y
        self.d = (1,1)
    
    def physics_loss(self, x, y, j):
        
        physics = j - torch.cos(self.w*x)
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return y, j
    
    def boundary_condition(self, x, y, j, sd):
        
        y, j = boundary_conditions.A_1D_1(x, y, j, self.A, 0, sd)
        return y, j

    def exact_solution(self, x, batch_size):
        
        Ap = self.A
        y = (1/self.w)*torch.sin(self.w*x) + Ap
        j = torch.cos(self.w*x)
        return y, j
    

class Cos_multi1D_1(_Problem):
    """Solves the 1D ODE:
        du
        -- = w1*cos(w1x) + w2*cos(w2x)
        dx
        
        Boundary conditions:
        u (0) = A
    """
    
    @property
    def name(self):
        return "Cos_multi1D_1_w%sw%s"%(self.w1, self.w2)
    
    def __init__(self, w1, w2, A=0):
        
        # input params
        self.w1, self.w2 = w1, w2
        
        # boundary params
        self.A = A
        
        # dimensionality of x and y
        self.d = (1,1)
    
    def physics_loss(self, x, y, j):
        
        physics = j - (self.w1*torch.cos(self.w1*x) + self.w2*torch.cos(self.w2*x))
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return y, j
    
    def boundary_condition(self, x, y, j, sd):
        
        y, j = boundary_conditions.A_1D_1(x, y, j, self.A, 0, sd)
        return y, j

    def exact_solution(self, x, batch_size):
        
        Ap = self.A
        y = torch.sin(self.w1*x) + torch.sin(self.w2*x) + Ap
        j = self.w1*torch.cos(self.w1*x) + self.w2*torch.cos(self.w2*x)
        return y, j

    
class Sin1D_2(_Problem):
    """Solves the 1D ODE:
        d^2 u
        ----- = sin(wx)
        dx^2
        
        Boundary conditions:
        u (0) = A
        u'(0) = B
    """
    
    @property
    def name(self):
        return "Sin1D_2_w%s"%(self.w)
    
    def __init__(self, w, A=0, B=0):
        
        # input params
        self.w = w
        
        # boundary params
        self.A = A
        self.B = B

        # dimensionality of x and y
        self.d = (1,1)
        
    def physics_loss(self, x, y, j, jj):
        
        physics = jj - torch.sin(self.w*x)
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jj = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]
        return y, j, jj
    
    def boundary_condition(self, x, y, j, jj, sd):
        
        y, j, jj = boundary_conditions.AB_1D_2(x, y, j, jj, self.A, self.B, 0, sd)
        return y, j, jj
    
    def exact_solution(self, x, batch_size):
        
        Ap = self.A
        Bp = self.B + (1/self.w)
        y = -(1/self.w**2)*torch.sin(self.w*x) + Bp*x + Ap
        j = -(1/self.w)   *torch.cos(self.w*x) + Bp
        jj = torch.sin(self.w*x)
        return y, j, jj


# 2D problems

class Cos_Cos2D_1(_Problem): #Not an ODE, because u is multivariate
    """Solves the 2D PDE:
        du   du
        -- + -- = cos(wx) + cos(wy)
        dx   dy
        
        Boundary conditions:
        u(0,y) = (1/w)sin(wy) + A

        my note: the solution is unique. consider two solutions u1, u2, let v=u1-u2
        then dv/dx + dv/dy = 0, v(0,y) = 0
        consider f(t) = v(0+t, y+t), f'(t) = dv/dx + dv/vy = 0, f(0) = 0, thus foreach t, f(t) = 0
        thus foreach x,y , v(x,y) = 0, thus u1 equals u2
    """
    
    @property
    def name(self):
        return "Cos_Cos2D_1_w%s"%(self.w)
    
    def __init__(self, w, A=0):
        
        # input params
        self.w = w
        
        # boundary params
        self.A = A
        
        # dimensionality of x and y
        self.d = (2,1)
    
    def physics_loss(self, x, y, j0, j1):
        
        physics = (j0[:,0] + j1[:,0]) - (torch.cos(self.w*x[:,0]) + torch.cos(self.w*x[:,1]))# be careful to slice correctly (transposed calculations otherwise (!))
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        
        return y, j0, j1
    
    def boundary_condition(self, x, y, j0, j1, sd):
        
        # Apply u = tanh((x-0)/sd)*NN + A + (1/w)sinwy   ansatz
        
        A, w = self.A, self.w
        
        t0, jt0 = boundary_conditions.tanh_1(x[:,0:1], 0, sd) #tanh(w*x_1), d/dx_1 tanh(w*x_1)
        
        sin = (1/w)*torch.sin(w*x[:,1:2])
        cos =       torch.cos(w*x[:,1:2])
        
        y_new  = t0 *y         + A + sin
        j0_new = jt0*y + t0*j0 #du/dx_1
        j1_new = t0 *j1            + cos #du/dx_2
        
        return y_new, j0_new, j1_new
    
    def exact_solution(self, x, batch_size):
        
        Ap = self.A
        y = (1/self.w)*torch.sin(self.w*x[:,0:1]) + (1/self.w)*torch.sin(self.w*x[:,1:2]) + Ap
        j0 = torch.cos(self.w*x[:,0:1])
        j1 = torch.cos(self.w*x[:,1:2])
        return y, j0, j1
    
    
class Sin2D_1(_Problem):
    """Solves the 2D PDE:
        du   du
        -- + -- = -sin(w(x+y))
        dx   dy
        
        Boundary conditions:
        u(x,x) = (1/w)cos^2(wx) + A
    """
    
    @property
    def name(self):
        return "Sin2D_1_w%s"%(self.w)
    
    def __init__(self, w, A=0):
        
        # input params
        self.w = w
        
        # boundary params
        self.A = A
        
        # dimensionality of x and y
        self.d = (2,1)
    
    def physics_loss(self, x, y, j0, j1):
        
        physics = (j0[:,0] + j1[:,0]) + (torch.sin(self.w*(x[:,0]+x[:,1])))# be careful to slice correctly (transposed calculations otherwise (!))
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        
        j =  torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:,0:1], j[:,1:2]
        
        return y, j0, j1
    
    def boundary_condition(self, x, y, j0, j1, sd):
        
        # Apply u = tanh((x+y)/sd)*NN + A + (1/w)cos^2wx   ansatz
        
        A, w = self.A, self.w
        
        t, jt = boundary_conditions.tanh_1(x[:,0:1]+x[:,1:2], 0, sd)
        
        cos2 = (1/w)*torch.cos(w*x[:,0:1])**2
        sin2 =    -2*torch.sin(w*x[:,0:1])*torch.cos(w*x[:,0:1])
        
        y_new  = t *y        + A + cos2
        j0_new = jt*y + t*j0     + sin2
        j1_new = jt*y + t*j1
        
        return y_new, j0_new, j1_new
    
    def exact_solution(self, x, batch_size):
        
        Ap = self.A
        y = (1/self.w)*torch.cos(self.w*x[:,0:1])*torch.cos(self.w*x[:,1:2]) + Ap
        j0 = -torch.sin(self.w*x[:,0:1])*torch.cos(self.w*x[:,1:2])
        j1 = -torch.cos(self.w*x[:,0:1])*torch.sin(self.w*x[:,1:2])
        return y, j0, j1


# 2x2 D problems

class Sin2x2D(_Problem):
    """Solves the 2x2D Problem:
        Domain:
        [0, 2*pi]x[0, 2*pi]

        Exact solution:
        u(x,y) = sin(x)
        v(x,y) = sin(y)

        Eqn:
        du/dx = cos(x)
        dv/dy = cos(y)
        
        Boundary conditions:
        u (0,y) = 0
        v (x,0) = 0

        Ansatz:
        u = tanh((x-mu)/sd) NN(x,y)
        v = tanh((y-mu)/sd) NN(x,y)
    """
    
    @property
    def name(self):
        return "Sin2x2D"
    
    def __init__(self):
        # dimensionality of x and y
        self.d = (2,2)
    
    def physics_loss(self, x, y, ju_0, jv_1):
        
        physics_one = ju_0 - torch.cos(x[:,0:1])
        physics_two = jv_1 - torch.cos(x[:,1:2])
        physics = torch.concat((physics_one, physics_two), dim=1)
        return losses.l2_loss(physics, 0)
    
    def get_gradients(self, x, y):
        
        j_u = torch.autograd.grad(y[:,0], x, torch.ones_like(y[:,0]), create_graph=True)[0]
        j_v = torch.autograd.grad(y[:,1], x, torch.ones_like(y[:,1]), create_graph=True)[0]
        ju_0 = j_u[:,0:1]
        jv_1 = j_v[:,1:2]
        return y, ju_0, jv_1
    
    def boundary_condition(self, x, y, ju_0, jv_1, sd):
        u, v = y[:,0:1], y[:,1:2]
        tu, jtu0 = boundary_conditions.tanh_1(x[:,0:1], 0, sd)
        tv, jtv1 = boundary_conditions.tanh_1(x[:,1:2], 0, sd)
        u_new = tu * u
        v_new = tv * v
        ju_0_new = jtu0 * u + tu * ju_0
        jv_1_new = jtv1 * v + tv * jv_1
        return torch.concat((u_new, v_new),dim=1), ju_0_new, jv_1_new

    def exact_solution(self, x, batch_size):
        
        y = torch.sin(x) # equal to torch.concat((torch.sin(x[:,0:1]), torch.sin(x[:,1:2])),dim=1)
        ju_0 = torch.cos(x[:,0:1])
        jv_1 = torch.cos(x[:,1:2])
        return y, ju_0, jv_1


class CavityFlow(_Problem):
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
    u(x,y) = tanh(x)tanh(x-1)tanh(y)tanh(y-1)*NN(x,y)[0] + 0.5*( 1 + tanh(20(y-0.9)) ) #an approximation
    v(x,y) = tanh(x)tanh(x-1)tanh(y)tanh(y-1)*NN(x,y)[1]
    """
    @property
    def name(self):
        return "Cavityflow2x3D"
    
    def __init__(self):
        # dimensionality of x and y
        # u, v, p = y[:,0], y[:,1], y[:,2]
        self.d = (2,3)
        self.re = 100
    
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
        u, v, p = y[:,0:1], y[:,1:2], y[:,2:3]
        tx, jtx, jjtx = boundary_conditions.tanhtanh_2(x[:,0:1], 0, 1, sd)
        ty, jty, jjty = boundary_conditions.tanhtanh_2(x[:,1:2], 0, 1, sd)
        tbd, jtbd, jjtbd = boundary_conditions.tanh_2(x[:,1:2], 0.9, 0.05)
        u_new = tx * ty * u + 0.5 * (tbd+1)
        u_new_x = ty * (jtx*u + tx*u_x)
        u_new_y = tx * (jty*u + ty*u_y) + 0.5*jtbd
        u_new_xx = ty * (jjtx*u + 2*jtx*u_x + tx*u_xx)
        u_new_yy = tx * (jjty*u + 2*jty*u_y + ty*u_yy) + 0.5*jjtbd
        v_new = tx * ty * v
        v_new_x = ty * (jtx*v + tx*v_x)
        v_new_y = tx * (jty*v + ty*v_y)
        v_new_xx = ty * (jjtx*v + 2*jtx*v_x + tx*v_xx)
        v_new_yy = tx * (jjty*v + 2*jty*v_y + ty*v_yy)
        y_new = torch.concat((u_new, v_new, p),dim=1)
        return y_new, u_new_x, u_new_y, u_new_xx, u_new_yy, v_new_x, v_new_y, v_new_xx, v_new_yy, p_x, p_y

    def exact_solution(self, x, batch_size):
        return (torch.zeros( (np.prod(batch_size),) + (3,) ), ) + (torch.zeros( (np.prod(batch_size),) + (1,) ), )*10 # shallow copy, but yj_true is read only



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    import problems
    from main import _x_mesh
    
    # check velocity models for WaveEquation3D
    
    P = problems.WaveEquation3D(c=1, source_sd=0.2)
    subdomain_xs = [np.array([-10, -5, 0, 5, 10]), np.array([-10, -5, 0, 5, 10]), np.array([0, 5, 10])]
    batch_size_test = (50,50,15)
    x = _x_mesh(subdomain_xs, batch_size_test, "cpu")
    
    for f in P._gaussian_c, P._constant_c:
        y = f(x)
        print(y.shape)
        y = y[:,0].numpy().reshape(batch_size_test)
        
        plt.figure()
        plt.imshow(y[:,:,0].T, origin="lower")
        plt.colorbar()
        plt.figure()
        plt.imshow(y[:,:,-1].T, origin="lower")
        plt.colorbar()
        plt.show()
    
    
