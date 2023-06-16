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

def pde(x, y, t, u):
    a = 20

    u_xx = VPINN.gradients(u, x, 2)
    u_yy = VPINN.gradients(u, y, 2)
    u_tt = VPINN.gradients(u, t, 2)

    return u_tt - (u_xx + a * a * u_yy)

def u(x, y, t):
    INITIAL_COEF_1 = 1
    INITIAL_COEF_2 = 1
    m1=1
    m2=1
    n1=1
    n2=1
    p1=1
    p2=1

    x = torch.cat([x, y, t], dim=1)

    return (
                INITIAL_COEF_1 * torch.sin(m1 * torch.pi * x[:, 0:1]) * torch.sinh(n1 * torch.pi * x[:, 1:2]) * torch.cos(p1 * torch.pi * x[:, 2:3])
                + INITIAL_COEF_2 * torch.sinh(m2 * torch.pi * x[:, 0:1]) * torch.sin(n2 * torch.pi * x[:, 1:2]) * torch.cos(p2 * torch.pi * x[:, 2:3])
            )

x1, x2, y1, y2, t1, t2 = (0, 1, 0, 1, 0, 100)
def bc(boundary_num=10):
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
    bc_ys = [yy, torch.full_like(xt, 0), torch.full_like(xt, 1), xt, xt]
    bc_ts = [torch.zeros_like(xx), tt, tt, tt, tt]
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = u(bc_xs, bc_ys, bc_ts)
    
    return (bc_xs, bc_ys, bc_ts, bc_us)

def wave_longtime(epoch=10000):
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 30, 30, 30, 1],pde, type='tanh', bc1=bc(100), area=[x1, x2, y1, y2, t1, t2], Q=10, grid_num=8, test_fcn_num=5, 
                device=device, load=None)


    net = vpinn.train('wave_longtime', epoch_num=epoch, coef=10)

    ################################################################################
    net.cpu()
    N = 100
    x = torch.linspace(x1, x2, N)
    y = torch.linspace(y1, y2, N)
    z = torch.linspace(t1, t2, N)

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = Z.reshape(-1, 1)

    prediction = net(torch.cat([X, Y, Z], dim=1))
    solution = u(X, Y, Z)
    # solution = u(xx, yy, zz)
    print(f'relative error={torch.norm(solution - prediction):.2f}/{torch.norm(solution):.2f}={torch.norm(solution - prediction) / torch.norm(solution) * 100:.2f}%')

    err = torch.norm(solution - prediction) / torch.norm(solution)
    torch.cuda.empty_cache()
    # return err
    return three_loss(prediction, solution)

if __name__ == "__main__":
    wave_longtime()
