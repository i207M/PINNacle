import torch
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from matplotlib.tri import Triangulation
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN2d import VPINN
from src.Loss import three_loss

x1, x2, y1, y2 = (0, 1, 0, 1)
bc = VPINN.complex_rec_bc(x1, x2, y1, y2, 100,\
        func1=lambda x, y: torch.cat([4 * x * (1 - x), torch.zeros_like(x), torch.zeros_like(x)], dim=1),\
        valid=[[True, True, False], [True, True, False], [False, False, True], [True, True, False]])

def ns_pde(x, y, u):
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

    momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
    momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy))
    continuity = u_vel_x + v_vel_y

    return torch.cat([momentum_x, momentum_y, continuity], dim=1)

def ns_lid_driven_flow(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([2, 20, 20, 20, 3], ns_pde, 'tanh', bc, area=[x1, x2, y1, y2],\
        Q=10, grid_num=4, test_fcn_num=5, device=device, load=None)
    
    net = vpinn.train('ns_lid_driven_flow', epoch_num=epoch, coef=0.1)
    net.cpu()
################################################################################
    # verify
    data = np.loadtxt('lid_driven.dat', skiprows=9)

    # get x、y、u of solution
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    p = data[:, 4]

    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)
    vv = torch.from_numpy(v).reshape(-1, 1).type(torch.float)
    pp = torch.from_numpy(p).reshape(-1, 1).type(torch.float)

    net.cpu()

    prediction = net(torch.cat([xx, yy], dim=1))
    solution = torch.cat([uu, vv, pp], dim=1)
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')

    # fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    # axes = axs.flatten()
    
    # x_ = torch.from_numpy(x__).reshape(-1, 1).to(torch.float32)
    # y_ = torch.from_numpy(y__).reshape(-1, 1).to(torch.float32)
    # t_ = torch.from_numpy(t__).reshape(-1, 1).to(torch.float32)
    # pred = net(torch.cat([x_, y_, t_], dim=1)).detach().numpy()
    
    
    # value = [u, prediction[:, 0].detach().numpy(), v, prediction[:, 1].detach().numpy(), p, prediction[:, 2].detach().numpy()]
    # tri = Triangulation(x, y)
    # name = ['u_real', 'u_pred', 'v_real', 'v_pred','p_real', 'p_pred']
    # for i in range(6):
    #     image = axes[i].tripcolor(tri, value[i], cmap='jet', edgecolors='k')
    #     axes[i].set_title(name[i])
    #     fig.colorbar(image, ax=axes[i])
    # plt.savefig('res.png')
    return three_loss(prediction, solution)

if __name__ == "__main__":
    ns_lid_driven_flow()