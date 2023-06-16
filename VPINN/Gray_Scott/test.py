import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from matplotlib.tri import Triangulation
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

def pde(x, y, t, value):
    u = value[:, 0:1]
    v = value[:, 0:1]
    # return VPINN.gradients(u, t, 1) - 1 / ((500 * torch.pi) ** 2) * VPINN.gradients(u, x, 2) - 1 / (torch.pi ** 2) * VPINN.gradients(u, y, 2)
    b = 0.04
    d = 0.1
    eps1 = 1e-5
    eps2 = 5e-5
    u_xx = VPINN.gradients(u, x, 2)
    u_yy = VPINN.gradients(u, y, 2)
    u_t = VPINN.gradients(u, t, 2)
    v_xx = VPINN.gradients(v, x, 2)
    v_yy = VPINN.gradients(v, y, 2)
    v_t = VPINN.gradients(v, t, 2)

    part1 = eps1 * (u_xx + u_yy) + b * (1 - u) - u * (v ** 2) - u_t
    part2 = eps2 * (v_xx + v_yy) - d * v + u * (v ** 2) - v_t

    return torch.cat([part1, part2], dim=1)
    
x1, x2, y1, y2, t1, t2 = (-1, 1, -1, 1, 0, 10)

def bc(boundary_num=10):
    x = torch.linspace(x1, x2, boundary_num)
    y = torch.linspace(y1, y2, boundary_num)
    t = torch.linspace(t1, t2, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xt, tt = torch.meshgrid(x, t, indexing='ij')
    
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xt = xt.reshape(-1, 1)
    tt = tt.reshape(-1, 1)

    bc_xs = xx
    bc_ys = yy
    bc_ts = torch.zeros_like(xx)
    
    bc_us = 1 - torch.exp(-80 * ((xx + 0.05)**2 + (yy + 0.02)**2))
    bc_vs = torch.exp(-80 * ((xx - 0.05)**2 + (yy - 0.02)**2))
    
    return (bc_xs, bc_ys, bc_ts, torch.cat([bc_us, bc_vs], dim=1))

def gray_scott(epoch=10000):
    torch.cuda.empty_cache()
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 30, 30, 30, 2],pde, type='tanh', bc1=bc(100), area=[x1, x2, y1, y2, t1, t2], Q=10, grid_num=4, test_fcn_num=5, 
                device=device, load=None)


    net = vpinn.train('gray_scott', epoch_num=epoch, coef=10)

    ################################################################################
    # verify
    data = np.loadtxt('grayscott.dat', skiprows=9)

    # get x、y、u of solution
    x = data[:, 0]
    y = data[:, 1]
    t = data[:, 2]
    u = data[:, 3]
    v = data[:, 4]

    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    
    tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)
    vv = torch.from_numpy(v).reshape(-1, 1).type(torch.float)

    net.cpu()

    prediction = net(torch.cat([xx, yy, tt], dim=1))
    solution = torch.cat([uu, vv], dim=1)
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')

    # return torch.norm(prediction - solution) / torch.norm(solution)
    return three_loss(prediction, solution)

if __name__ == "__main__":
    gray_scott()
