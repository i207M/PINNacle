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

x1, x2, y1, y2 = (0, 4, 0, 2)

def bc_frame(bc_num=100):
    x_down = torch.linspace(x1, x2, bc_num).reshape(-1, 1)
    y_down = torch.linspace(y1, y1, bc_num).reshape(-1, 1)
    u_down = torch.zeros_like(x_down)
    v_down = torch.zeros_like(u_down)
    p_down = v_down
    validation_down = torch.cat([torch.full_like(u_down, True, dtype=torch.bool), torch.full_like(v_down, True, dtype=torch.bool), torch.full_like(p_down, False, dtype=torch.bool)], dim=1)

    x_left = torch.linspace(x1, x1, bc_num).reshape(-1, 1)
    y_left = torch.linspace(y1, y2 / 2, bc_num).reshape(-1, 1)
    u_left = 4 * y_left * (1 - y_left)
    v_left = torch.zeros_like(u_left)
    p_left = v_left
    validation_left = torch.cat([torch.full_like(u_left, True, dtype=torch.bool), torch.full_like(v_left, True, dtype=torch.bool), torch.full_like(p_left, False, dtype=torch.bool)], dim=1)

    x_right = torch.linspace(x2, x2, bc_num).reshape(-1, 1)
    y_right = torch.linspace(y1, y2, bc_num).reshape(-1, 1)
    u_right = torch.zeros_like(x_right)
    v_right = torch.zeros_like(u_right)
    p_right = v_right
    validation_right = torch.cat([torch.full_like(u_right, False, dtype=torch.bool), torch.full_like(v_right, False, dtype=torch.bool), torch.full_like(p_right, True, dtype=torch.bool)], dim=1)

    x_up = torch.linspace(x1 + (x2 - x1) / 2, x2, bc_num).reshape(-1, 1)
    y_up = torch.linspace(y2, y2, bc_num).reshape(-1, 1)
    u_up = torch.zeros_like(x_up)
    v_up = torch.zeros_like(u_up)
    p_up = torch.zeros_like(x_up)
    validation_up = torch.cat([torch.full_like(u_up, True, dtype=torch.bool), torch.full_like(v_up, True, dtype=torch.bool), torch.full_like(p_up, False, dtype=torch.bool)], dim=1)

    x = torch.cat([x_down, x_left, x_right, x_up], dim=0)
    y = torch.cat([y_down, y_left, y_right, y_up], dim=0)
    u = torch.cat([u_down, u_left, u_right, u_up], dim=0)
    v = torch.cat([v_down, v_left, v_right, v_up], dim=0)
    p = torch.cat([p_down, p_left, p_right, p_up], dim=0)
    validation = torch.cat([validation_down, validation_left, validation_right, validation_up], dim=0)
    value = torch.cat([u, v, p], dim=1)

    return [x, y, value, validation]

def sample_points_on_circle(x, y, r, n=100):
    angles = torch.linspace(0, 2 * torch.pi, n)  # 在0到2π之间等间隔采样n个角度值
    x_coords = x + r * torch.cos(angles)  # 计算x坐标
    y_coords = y + r * torch.sin(angles)  # 计算y坐标

    return x_coords, y_coords


# bc_rec = VPINN.complex_rec_bc(x1, x2, y1, y2, num=100, \
#             func4=lambda x, y: torch.cat([4 * y * (1 - y), torch.zeros_like(x), torch.zeros_like(x)], dim=1),\
#             valid=[[True, True, False], [False, False, True], [True, True, False], [True, True, False]])

circles = [(1, 0.5, 0.2), (2, 0.5, 0.3), (2.7, 1.6, 0.2)]
x_circles = []
y_circles = []
u_circles = [[], [], []]
validation_circles = []

for circle in circles:
    xx, yy = sample_points_on_circle(circle[0], circle[1], circle[2], n=100)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    uu = torch.zeros_like(xx)
    vv = torch.zeros_like(xx)
    pp = torch.zeros_like(xx)
    validation_circle = torch.cat([torch.full_like(uu, True, dtype=torch.bool), torch.full_like(vv, True, dtype=torch.bool), torch.full_like(pp, False, dtype=torch.bool)], dim=1)
    x_circles.append(xx)
    y_circles.append(yy)
    u_circles[0].append(uu)
    u_circles[1].append(vv)
    u_circles[2].append(pp)
    validation_circles.append(validation_circle)
x_circles = torch.cat(x_circles, dim=0)
y_circles = torch.cat(y_circles, dim=0)
validation_circles = torch.cat(validation_circles, dim=0)
for i in range(3):
    u_circles[i] = torch.cat(u_circles[i], dim=0)
u_circles = torch.cat(u_circles, dim=1)

bc_circles = [x_circles, y_circles, u_circles, validation_circles]

bc_obstacle_rec = VPINN.complex_rec_bc(2.8, 3.6, 0.8, 1.1, num=100, \
            func4=lambda x, y: torch.cat([torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=1),\
            valid=[[True, True, False], [True, True, False], [True, True, False], [True, True, False]])

bc = [0, 0, 0, 0]
for i in range(4):
    bc[i] = torch.cat([bc_frame()[i], bc_circles[i], bc_obstacle_rec[i]], dim=0)

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

# bc = dirichlet_bc()

# x = bc[0]
# y = bc[1]
# u = bc[2]
# validation = bc[3]

# # 合并 x, y 和 u
# data = torch.cat([x, y, u, validation], dim=1)

# np.savetxt('bc.txt', data.numpy(), fmt='%.4f', delimiter='\t')

def ns_2d_back_step_flow(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([2, 20, 20, 20, 3], ns_pde, 'tanh', bc, area=[x1, x2, y1, y2],\
        Q=10, grid_num=8, test_fcn_num=5, device=device, load=None)
    
    net = vpinn.train('2d_back_step_flow', epoch_num=epoch, coef=1)
    net.cpu()
################################################################################
    # verify
    data = np.loadtxt('ns_4_obstacle.dat', skiprows=9)

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

    # fig, axs = plt.subplots(3, 2, figsize=(13, 9))
    # axes = axs.flatten()
    
    # value = [u, prediction[:, 0].detach().numpy(), v, prediction[:, 1].detach().numpy(), p, prediction[:, 2].detach().numpy()]
    # tri = Triangulation(x, y)
    # name = ['u_real', 'u_pred', 'v_real', 'v_pred','p_read', 'p_pred']
    # for i in range(6):
    #     image = axes[i].scatter(x, y, c=value[i], cmap='viridis') # 'viridis' 是一种颜色映射，你也可以选择其他的
    #     # plt.colorbar() 
    #     # image = axes[i].tripcolor(tri, value[i], cmap='jet', edgecolors='k')
    #     axes[i].set_title(name[i])
    #     fig.colorbar(image, ax=axes[i])
    # plt.savefig('res.png')
    return three_loss(prediction, solution)

if __name__ == "__main__":
    ns_2d_back_step_flow()