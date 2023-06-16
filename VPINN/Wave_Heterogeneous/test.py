import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from matplotlib.tri import Triangulation
from pathlib import Path
from scipy import interpolate
from .func_cache import cache_tensor
from src.Loss import three_loss
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN

script_path = os.path.abspath(__file__)

# 获取脚本所在的目录
script_dir = os.path.dirname(script_path)

# 切换到脚本所在的目录
os.chdir(script_dir)
darcy_2d_coef = np.loadtxt("darcy_2d_coef_256.dat")
@cache_tensor
def coef(x):
    return torch.Tensor(
        interpolate.griddata(darcy_2d_coef[:, 0:2], darcy_2d_coef[:, 2], (x.detach().cpu().numpy()[:, 0:2] + 1) / 2)
    ).unsqueeze(dim=-1)

def wave_pde(x, y, t, u, device='cuda:3'):
    u_xx = VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2)
    u_tt = VPINN.gradients(u, t, 2)

    return u_xx - u_tt / coef(torch.cat([x, y, t], dim=1)).to(device)

x1, x2, y1, y2, t1, t2 = (-1, 1, -1, 1, 0 ,5)

def bc(boundary_num=10):
    mu = (-0.5 , 0)
    sigma = 0.3

    x = torch.linspace(x1, x2, boundary_num)
    y = torch.linspace(y1, y2, boundary_num)
    t = torch.linspace(t1, t2, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xt, tt = torch.meshgrid(x, t, indexing='ij')
    
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xt = xt.reshape(-1, 1)
    tt = tt.reshape(-1, 1)

    bc_xs = [xx]
    bc_ys = [yy]
    bc_ts = [torch.zeros_like(xx)]
    bc_us = [torch.exp(-((xx - mu[0]) ** 2 + (yy - mu[1]) ** 2)/ (2 * sigma ** 2))]
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = torch.cat(bc_us, dim=0).reshape(-1, 1)
    
    return (bc_xs, bc_ys, bc_ts, bc_us)
 
def neumann_operation(x, y, t, u):
    grad_u_x = VPINN.gradients(u, x)
    grad_u_y = VPINN.gradients(u, y)
    
    # 初始化结果张量
    grad_u_n = torch.zeros_like(x)
    
    # 对每个边界进行处理
    x_boundary = torch.where((x <= x1) | (x >= x2))
    y_boundary = torch.where((y <= y1) | (y >= y2))

    # 根据位置设置边界条件
    grad_u_n[x_boundary] = grad_u_x[x_boundary]
    grad_u_n[y_boundary] = grad_u_y[y_boundary]

    return grad_u_n
        

def bc_neumann(boundary_num=10):
    mu = (-0.5 , 0)
    sigma = 0.3

    x = torch.linspace(x1, x2, boundary_num)
    y = torch.linspace(y1, y2, boundary_num)
    t = torch.linspace(t1, t2, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xt, tt = torch.meshgrid(x, t, indexing='ij')
    
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xt = xt.reshape(-1, 1)
    tt = tt.reshape(-1, 1)

    bc_xs = [xt, xt, torch.full_like(xt, 0), torch.full_like(xt, 1)]
    bc_ys = [torch.full_like(xt, 0), torch.full_like(xt, 1), xt, xt]
    bc_ts = [tt, tt, tt, tt]
    bc_us = [torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0)]
    operation = neumann_operation
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = torch.cat(bc_us, dim=0).reshape(-1, 1)
    return [bc_xs, bc_ys, bc_ts, bc_us, operation]

def wave_heterogeneous(epoch=10000):
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 30, 30, 30, 1], wave_pde, type='tanh', bc1=bc(20), bc2=bc_neumann(20), area=[x1, x2, y1, y2, t1, t2], Q=10, grid_num=2, test_fcn_num=5, 
                device=device, load=None)

    # profiler=Profiler()
    # profiler.start()

    net = vpinn.train('wave_heterogeneous', epoch_num=epoch, coef=10)

    # profiler.stop()
    # profiler.print()
    # net = vpinn.train(None, epoch_num=0, coef=0.1)

    ################################################################################
    # verify
    data = np.loadtxt('wave_darcy.dat', skiprows=9)

    # get x、y、u of solution
    x_ = data[:, 0]
    y_ = data[:, 1]
    x = []
    y = []
    t = []
    u = []
    for i in range(251):
        x.append(copy.deepcopy(x_))
        y.append(copy.deepcopy(y_))
        t.append([i * 0.02 for _ in range(len(x_))])
        u.append(data[:, i + 2])

    x = np.concatenate(x)
    y = np.concatenate(y)
    t = np.concatenate(t)
    u = np.concatenate(u)
    # tri = Triangulation(x, t)

    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

    net.cpu()

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
    # torch.cuda.empty_cache()
    # return err
    return three_loss(prediction, uu)

if __name__ == "__main__":
    wave_heterogeneous()
