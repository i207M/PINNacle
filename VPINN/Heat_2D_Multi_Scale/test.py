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
    return VPINN.gradients(u, t, 1) - 1 / ((500 * torch.pi) ** 2) * VPINN.gradients(u, x, 2) - 1 / (torch.pi ** 2) * VPINN.gradients(u, y, 2)

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
    
    ######################################################################
    # data = np.loadtxt('heat_multiscale_lesspoints.dat', skiprows=9)

    # # get x、y、u of solution
    # x_ = data[:, 0]
    # y_ = data[:, 1]
    # x = []
    # y = []
    # t = []
    # u = []
    # for i in range(26):
    #     x.append(copy.deepcopy(x_))
    #     y.append(copy.deepcopy(y_))
    #     t.append([i * 0.2 for _ in range(len(x_))])
    #     u.append(data[:, i + 2])

    # x = np.concatenate(x)
    # y = np.concatenate(y)
    # t = np.concatenate(t)
    # u = np.concatenate(u)
    # tri = Triangulation(x, t)

    # xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    # yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    # tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
    # uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

    # x_initial = xx[0:3000]
    # y_initial = yy[0:3000]
    # t_initial = tt[0:3000]
    # u_initial = uu[0:3000]
    # bc_xs = [x_initial]
    # bc_ys = [y_initial]
    # bc_ts = [t_initial]
    # bc_us = [u_initial]
    #########################################################################
    bc_xs = [xx, xt, xt, torch.full_like(xt, 0), torch.full_like(xt, 1)]
    bc_ys = [yy, torch.full_like(xt, 0), torch.full_like(xt, 1), xt, xt]
    bc_ts = [torch.zeros_like(xx), tt, tt, tt, tt]
    bc_us = [torch.sin(20 * torch.pi * xx) * torch.sin(torch.pi * yy), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0)]
    
    # bc_xs = [xx]
    # bc_ys = [yy]
    # bc_ts = [torch.zeros_like(xx)]
    # bc_us = [torch.sin(20 * torch.pi * xx) * torch.sin(torch.pi * yy)]
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = torch.cat(bc_us, dim=0).reshape(-1, 1)
    
    return (bc_xs, bc_ys, bc_ts, bc_us)

def heat_2d_multi_scale(epoch=10000):
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 30, 30, 30, 1],pde, type='tanh', bc1=bc(100), area=[0, 1, 0, 1, 0, 5], Q=10, grid_num=8, test_fcn_num=5, 
                device=device, load=None)

    # profiler=Profiler()
    # profiler.start()

    net = vpinn.train('heat_2d_multi_scale', epoch_num=epoch, coef=50)

    # profiler.stop()
    # profiler.print()
    # net = vpinn.train(None, epoch_num=0, coef=0.1)

    ################################################################################
    # verify
    data = np.loadtxt('heat_multiscale_lesspoints.dat', skiprows=9)

    # get x、y、u of solution
    x_ = data[:, 0]
    y_ = data[:, 1]
    x = []
    y = []
    t = []
    u = []
    for i in range(26):
        x.append(copy.deepcopy(x_))
        y.append(copy.deepcopy(y_))
        t.append([i * 0.2 for _ in range(len(x_))])
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

    x_initial = xx[0:3000]
    y_initial = yy[0:3000]
    t_initial = tt[0:3000]
    u_initial = uu[0:3000]
    net.cpu()
    sol = torch.sin(20 * torch.pi * x_initial) * torch.sin(torch.pi * y_initial)
    pred = net(torch.cat([x_initial, y_initial, t_initial], dim=1))
    diff = pred - u_initial
    # x = torch.linspace(-1, 1, 50)
    # y = x
    # z = x
    # xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    # xx = xx.reshape(-1, 1)
    # yy = yy.reshape(-1, 1)
    # zz = zz.reshape(-1, 1)

    prediction = net(torch.cat([xx, yy, tt], dim=1))
    # solution = u(xx, yy, zz)
    print(f'relative error={torch.norm(prediction - uu):.2f}/{torch.norm(uu):.2f}={torch.norm(prediction - uu) / torch.norm(uu) * 100:.2f}%')

    err = torch.norm(prediction - uu) / torch.norm(uu)
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # axes = axs.flatten()
    # for i in range(6):
    #     # z = torch.tensor(-1 + 0.5 * i)
    #     # zz = torch.full_like(xx, -1 + i * 0.5)
    #     # zz = zz.reshape(-1, 1)

    #     xx = xx[0:3000]
    #     yy = yy[0:3000]
    #     prediction = net(torch.cat([xx, yy, tt[3000 * (5 * i): 3000 * (5 * i + 1)]], dim=1)).detach()
    #     solution = uu[3000 * (5 * i): 3000 * (5 * i + 1)]
    #     res = (prediction - solution).reshape(-1).detach().numpy()
            
    #     tri = Triangulation(xx.reshape(-1), yy.reshape(-1))

    #     image = axes[i].tripcolor(tri, res, cmap='jet', edgecolors='k')
    #     axes[i].set_title(f'z={i}')
    #     fig.colorbar(image, ax=axes[i])
    # plt.savefig('res.png')
    
    return three_loss(prediction, uu)
    return err

if __name__ == "__main__":
    heat_2d_multi_scale()
