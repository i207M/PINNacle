import torch
import numpy as np
import sys, os
# import pyinstrument
import copy
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import interpolate
from matplotlib.tri import Triangulation

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

#############################################################################################
# define the pde
def f(x, y, t):
    A = 200
    m1 = 1
    m2 = 5
    m3 = 1
    sin = torch.sin
    pi = torch.pi
    return A * sin(m1 * pi * x) * sin(m2 * pi * y) * sin(m3 * pi * t)

# VPINN.laplace represents the result of laplace operator applied to u with Green Theorem
# When VPINN.laplace is used, you are expected to wrap the monomial with VPINN.LAPLACE_TERM() to distinguish
# VPINN.LAPLACE_TERM can only contain a monomial of VPINN.laplace, polynomial is not allowed

def coef(x, y, t):
    heat_2d_coef = np.loadtxt("heat_2d_coef_256.dat")
    X = torch.cat([x, y, t], dim=1)
    return torch.Tensor(
                interpolate.griddata(heat_2d_coef[:, 0:2], heat_2d_coef[:, 2], X.detach().cpu().numpy()[:, 0:2], method="nearest")
            )
def pde(x, y, t, u, device='cuda'):    
    return VPINN.gradients(u, t, 1) - coef(x, y, t).to(device).reshape(-1, 1) * (VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2)) - f(x, y, t)

# this pde doesn't use the green theorem to simplify the equation
# def pde(x, y, u):    
    # return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) - f(x, y)

#############################################################################################
# boundary condition
def bc(boundary_num):
    x = torch.linspace(0, 1, boundary_num)
    y = torch.linspace(0, 1, boundary_num)
    t = torch.linspace(0, 5, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xt, tt = torch.meshgrid(x, t, indexing='ij')
    
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xt = xt.reshape(-1, 1)
    tt = tt.reshape(-1, 1)
    
    bc_xs = [xx, xt, xt, torch.full_like(xt, 0), torch.full_like(xt, 1)]
    bc_ys = [yy, torch.full_like(xx, 0), torch.full_like(xt, 1), xt, xt]
    bc_ts = [torch.zeros_like(xx), tt, tt, tt, tt]
    bc_us = [torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0)]
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = torch.cat(bc_us, dim=0).reshape(-1, 1)
    return [bc_xs, bc_ys, bc_ts, bc_us]

#############################################################################################
# train the model
def heat_varying_source(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 10, 10, 10, 1],pde, bc1=bc(20), area=[0, 1, 0, 1, 0, 5], Q=10, grid_num=4, test_fcn_num=5, 
                device=device, load=None)

    net = vpinn.train('heat_darcy', epoch_num=epoch, coef=10)
    net.cpu()
    #############################################################################################
    # plot and verify
    data = np.loadtxt('heat_darcy.dat', skiprows=9)

    # get x、y、u of solution
    x_ = data[:, 0]
    y_ = data[:, 1]
    x = []
    y = []
    t = []
    u = []
    for i in range(51):
        x.append(copy.deepcopy(x_))
        y.append(copy.deepcopy(y_))
        t.append([i * 0.1 for _ in range(len(x_))])
        u.append(data[:, i + 2])

    x = np.concatenate(x)
    y = np.concatenate(y)
    t = np.concatenate(t)
    u = np.concatenate(u)
    tri = Triangulation(x, t)

    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

    prediction = net(torch.cat([xx, yy, tt], dim=1))
    # solution = u(xx, yy, zz)
    print(f'relative error={torch.norm(prediction - uu):.2f}/{torch.norm(uu):.2f}={torch.norm(prediction - uu) / torch.norm(uu) * 100:.2f}%')

    err = torch.norm(prediction - uu) / torch.norm(uu)
    return three_loss(prediction, uu)
    
if __name__ == "__main__":
    heat_varying_source()