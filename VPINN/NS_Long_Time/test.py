import torch
import copy
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from matplotlib.tri import Triangulation
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

x1, x2, y1, y2, t1, t2 = (0, 2, 0, 1, 0, 5)
A1, A2, A3 = (1, 1, 1)
sin = torch.sin
pi = torch.pi
bc = VPINN.complex_cube_bc(x1, x2, y1, y2, t1, t2, 100,\
        func6=lambda x, y, t: torch.cat([sin(pi *y) * (A1 * sin((pi * t) + A2 * sin(3 * pi * t) + A3 * sin(5 * pi *t))), torch.zeros_like(x), torch.zeros_like(x)], dim=1),\
        valid=[[True, True, False], [True, True, False], [False, False, True], [True, True, False], [True, True, False], [True, True, False]])

def initial_f(x, y, t):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y) * torch.sin(np.pi * t)

def ns_pde(x, y, t, u):
    nu = 1 / 100
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:]
    u_vel_x = VPINN.gradients(u_vel, x, 1)
    u_vel_y = VPINN.gradients(u_vel, y, 1)
    u_vel_xx = VPINN.gradients(u_vel, x, 2)
    u_vel_yy = VPINN.gradients(u_vel, y, 2)

    v_vel_x = VPINN.gradients(v_vel, x, 1)
    v_vel_y = VPINN.gradients(v_vel, y, 1)
    v_vel_xx = VPINN.gradients(v_vel, x, 2)
    v_vel_yy = VPINN.gradients(v_vel, y, 2)

    p_x = VPINN.gradients(p, x, 1)
    p_y = VPINN.gradients(p, y, 1)

    u_t = VPINN.gradients(u_vel, t, 1)
    v_t = VPINN.gradients(v_vel, t, 1)

    momentum_x = (u_t + u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy) + initial_f(x, y, t))
    momentum_y = (v_t + u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy) + initial_f(x, y, t))
    continuity = u_vel_x + v_vel_y

    return torch.cat([momentum_x, momentum_y, continuity], dim=1)

def ns_longtime(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 20, 20, 20, 3], ns_pde, 'tanh', bc, area=[x1, x2, y1, y2, t1, t2],\
        Q=10, grid_num=4, test_fcn_num=5, device=device, load=None)
    
    net = vpinn.train('ns_longtime', epoch_num=epoch, coef=0.1)
    net.cpu()
################################################################################
    # verify
    data = np.loadtxt('ns_long.dat', skiprows=9)

    # # get x、y、u of solution
    # x = data[:, 0]
    # y = data[:, 1]
    # u = data[:, 2]
    # v = data[:, 3]
    # p = data[:, 4]

    # xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    # yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    
    # uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)
    # vv = torch.from_numpy(v).reshape(-1, 1).type(torch.float)
    # pp = torch.from_numpy(p).reshape(-1, 1).type(torch.float)

    # net.cpu()
    # get x、y、u of solution
    x_ = data[:, 0]
    y_ = data[:, 1]
    x = []
    y = []
    t = []
    u = []
    v = []
    p = []
    for i in range(51):
        x.append(copy.deepcopy(x_))
        y.append(copy.deepcopy(y_))
        t.append([i * 0.1 for _ in range(len(x_))])
        u.append(data[:, 3 * i + 2])
        v.append(data[:, 3 * i + 3])
        p.append(data[:, 3 * i + 4])

    x = np.concatenate(x)
    y = np.concatenate(y)
    t = np.concatenate(t)
    u = np.concatenate(u)
    v = np.concatenate(v)
    p = np.concatenate(p)
    # tri = Triangulation(x, t)

    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)
    vv = torch.from_numpy(v).reshape(-1, 1).type(torch.float)
    pp = torch.from_numpy(p).reshape(-1, 1).type(torch.float)


    net.cpu()

    prediction = net(torch.cat([xx, yy, tt], dim=1))
    solution = torch.cat([xx, yy, pp], dim=1)
    # solution = u(xx, yy, zz)
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')

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
    torch.cuda.empty_cache()
    return three_loss(prediction, solution)

    prediction = net(torch.cat([xx, yy], dim=1))
    solution = torch.cat([uu, vv, pp], dim=1)
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')

if __name__ == "__main__":
    ns_longtime()