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


class Kuramoto(_Problem):
    """
    Solves the Kuramoto-Sivashinsky equation
    u(x,0) = cos(x) (1+sin(x))
    Ansatz: u(x,t) = u(x,0) + NN(x,t)tanh(t)
    """
    @property
    def name(self):
        return "Kuramoto"

    def __init__(self):
        self.bbox=[0, 2 * np.pi, 0, 1]
        self.d = (2,1) # x,t
        self.load_ref_data("Kuramoto_Sivashinsky")
        # scale t (let t1 = t_scale * t, solve new eqn w.r.t. t1 and plot w.r.t t1)
        self.t_scale = 5
        self.ref_x[:,1] = self.t_scale * self.ref_x[:,1]
        self.bbox[-1] = self.bbox[-1] * self.t_scale
        # prepare for interpn
        t_nums = 251
        xs = self.ref_x[::t_nums,0]
        ts = self.ref_x[:t_nums,1]
        self.ref_in_coords = (xs, ts)
        self.ref_values = (self.ref_y.reshape(-1, t_nums),)
        # downsample randomly on ref solution to avoid using too much memory
        self.downsample_ref_data(6)
        self.num_js = 4

        self.alpha = 100/16
        self.beta = 100/16**2
        self.gamma = 100/16**4
    
    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jx, jt = j[:,0:1], j[:,1:2]
        jxx = torch.autograd.grad(jx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jxxx = torch.autograd.grad(jxx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        jxxxx = torch.autograd.grad(jxxx, x, torch.ones_like(jx), create_graph=True)[0][:,0:1]
        return y, jt, jx, jxx, jxxxx

    def physics_loss(self, x, y, jt, jx, jxx, jxxxx):
        physics = self.t_scale*jt[:,0] + self.alpha*y[:,0]*jx[:,0] + self.beta*jxx[:,0] + self.gamma*jxxxx[:,0]
        return losses.l2_loss(physics, 0)
    
    def boundary_condition(self, x, y, jt, jx, jxx, jxxxx, sd):
        t0, jt0 = boundary_conditions.tanh_1(x[:,1:2], 0, sd)
        u0 = torch.cos(x[:,0:1]) * (1 + torch.sin(x[:,0:1]))
        u0x = -torch.sin(x[:,0:1]) + torch.cos(2 * x[:,0:1])
        u0xx = -torch.cos(x[:,0:1]) - 2*torch.sin(2 * x[:,0:1])
        u0xxxx = torch.cos(x[:,0:1]) + 8 * torch.sin(2 * x[:,0:1])
        y_new = u0 + y * t0
        jt_new = jt * t0 + y * jt0
        jx_new = u0x + jx * t0
        jxx_new = u0xx + jxx * t0
        jxxxx_new = u0xxxx + jxxxx * t0
        return y_new, jt_new, jx_new, jxx_new, jxxxx_new


class GrayScott(_Problem):
    """
    Solves the 2d diffusion reaction Gray-Scott model
    u and v represent concentrations of chemicals
    Initial Condition: u(x,y,0), v(x,y,0)
    Ansatz: {u,v} = NN(x,y,t) tanh(t) + {u,v}(x,y,0)
    """
    @property
    def name(self):
        return "GrayScott"
    
    def __init__(self):
        self.bbox=[-1, 1, -1, 1, 0, 200]
        self.d = (3,2) # x,y,t -> u,v
        self.load_ref_data("grayscott") # order: x,y,t; u,v
        #self.ref_x = self.ref_x.reshape(21,10000,3)[::5,:,:].reshape(-1,3)
        #self.ref_y = self.ref_y.reshape(21,10000,2)[::5,:,:].reshape(-1,2)
        # prepare for interpn
        xs = self.ref_x[:100, 0]
        ys = self.ref_x[:10000:100, 1]
        ts = self.ref_x[::10000, 2]
        self.ref_in_coords = (xs, ys, ts)
        us = np.transpose(self.ref_y[:, 0].reshape(21, 100, 100), [2,1,0])
        vs = np.transpose(self.ref_y[:, 1].reshape(21, 100, 100), [2,1,0])
        self.ref_values = (us, vs)
        self.downsample_ref_data(6)
        self.num_js = 4

        self.b_param = 0.04
        self.d_param = 0.1
        self.expdecay = 80
        self.eps1 = 1e-5
        self.eps2 = 5e-6

    def get_gradients(self, x, y):
        uj = torch.autograd.grad(y[:,0], x, torch.ones_like(y[:,0]), create_graph=True)[0]
        ujx, ujy, ujt = uj[:,0:1], uj[:,1:2], uj[:,2:3]
        vj = torch.autograd.grad(y[:,1], x, torch.ones_like(y[:,1]), create_graph=True)[0]
        vjx, vjy, vjt = vj[:,0:1], vj[:,1:2], vj[:,2:3]
        ujxx = torch.autograd.grad(ujx, x, torch.ones_like(ujx), create_graph=True)[0][:,0:1]
        ujyy = torch.autograd.grad(ujy, x, torch.ones_like(ujy), create_graph=True)[0][:,1:2]
        vjxx = torch.autograd.grad(vjx, x, torch.ones_like(vjx), create_graph=True)[0][:,0:1]
        vjyy = torch.autograd.grad(vjy, x, torch.ones_like(vjy), create_graph=True)[0][:,1:2]
        return y, ujt, vjt, (ujxx+ujyy), (vjxx+vjyy)

    def physics_loss(self, x, y, ujt, vjt, nabla_u, nabla_v):
        u, v = y[:,0:1], y[:,1:2]
        physics_u = self.eps1 * nabla_u + self.b_param * (1-u) - u*v*v - ujt
        physics_v = self.eps2 * nabla_v - self.d_param*v + u*v*v - vjt
        physics = torch.concat((physics_u, physics_v), dim=1)
        return losses.l2_loss(physics, 0)
    
    def boundary_condition(self, x, y, ujt, vjt, nabla_u, nabla_v, sd):
        u, v = y[:,0:1], y[:,1:2]
        inner_u = (x[:,0:1]+0.05)**2 + (x[:,1:2]+0.02)**2
        u0 = 1 - torch.exp(-self.expdecay * inner_u)
        inner_v = (x[:,0:1]-0.05)**2 + (x[:,1:2]-0.02)**2
        v0 = torch.exp(-self.expdecay * inner_v)
        nabla_u0 = - ( (-2*self.expdecay + self.expdecay**2*4*(x[:,0:1]+0.05)) + (-2*self.expdecay + self.expdecay**2*4*(x[:,1:2]+0.02)) ) * torch.exp(-self.expdecay * inner_u)
        nabla_v0 = ( (-2*self.expdecay + self.expdecay**2*4*(x[:,0:1]-0.05)) + (-2*self.expdecay + self.expdecay**2*4*(x[:,1:2]-0.02)) ) * torch.exp(-self.expdecay * inner_v)
        t0, jt0 = boundary_conditions.tanh_1(x[:,2:3], 0, sd)
        y_new = torch.concat((u0+u*t0, v0+v*t0), dim=1)
        ujt_new, vjt_new = ujt*t0 + u*jt0 , vjt*t0 + v*jt0
        nabla_u_new, nabla_v_new = nabla_u0 + nabla_u*t0, nabla_v0 + nabla_v*t0
        return y_new, ujt_new, vjt_new, nabla_u_new, nabla_v_new
