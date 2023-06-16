import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from matplotlib.tri import Triangulation
from pathlib import Path
from scipy.interpolate import griddata

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

x1, x2, y1, y2, z1, z2 = (0, 1, 0, 1, 0, 1)
circ=[(0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2), (0.2, 0.8, 0.7, 0.1), (0.6, 0.2, 0.3, 0.1)]

def neumann_conditions(x, y, z, u, x1, x2, y1, y2, z1, z2, z3):
    grad_u_x = VPINN.gradients(u, x)
    grad_u_y = VPINN.gradients(u, y)
    grad_u_z = VPINN.gradients(u, z)
    
    # 初始化结果张量
    grad_u_n = torch.zeros_like(x)
    
    # 对每个边界进行处理
    x_boundary = torch.where((x <= x1) | (x >= x2))
    y_boundary = torch.where((y <= y1) | (y >= y2))
    z_boundary1 = torch.where(z <= z1)
    z_boundary2 = torch.where((z > z1) & (z < z3))
    z_boundary3 = torch.where(z >= z3)

    # 根据位置设置边界条件
    grad_u_n[x_boundary] = grad_u_x[x_boundary]
    grad_u_n[y_boundary] = grad_u_y[y_boundary]
    grad_u_n[z_boundary1] = grad_u_z[z_boundary1]
    grad_u_n[z_boundary2] = grad_u_z[z_boundary2]  # 在z2处依然对z求导
    grad_u_n[z_boundary3] = grad_u_z[z_boundary3]

    return grad_u_n

def sphere_neumann_conditions(x, y, z, u, cx, cy, cz):
    grad_u_x = VPINN.gradients(u, x)
    grad_u_y = VPINN.gradients(u, y)
    grad_u_z = VPINN.gradients(u, z)
    
    # 计算球面的法向量
    normal_x = x - cx
    normal_y = y - cy
    normal_z = z - cz
    
    # 法向导数就是梯度与法线的点乘
    grad_u_n = grad_u_x * normal_x + grad_u_y * normal_y + grad_u_z * normal_z
    
    return grad_u_n

def sphere_bc(cx, cy, cz, r, num):
    # 在单位球上均匀分布的点
    u = torch.rand(num, 1)
    v = torch.rand(num, 1)
    
    theta = 2 * np.pi * u
    phi = torch.acos(2 * v - 1)
    
    # 将点映射到目标球上
    x = r * torch.sin(phi) * torch.cos(theta) + cx
    y = r * torch.sin(phi) * torch.sin(theta) + cy
    z = r * torch.cos(phi) + cz
    
    # 边界值设为 0
    bc_val = torch.zeros_like(x)
    
    operation = lambda x, y, z, u: sphere_neumann_conditions(x, y, z, u, cx, cy, cz)
    
    return [x, y, z, bc_val, operation]

def combined_neumann_conditions(x, y, z, u, x1, x2, y1, y2, z1, z2, z3, spheres):
    # 计算矩形的 Neumann 边界条件
    grad_u_n = neumann_conditions(x, y, z, u, x1, x2, y1, y2, z1, z2, z3)
    
    # 对每个球体，计算球面的 Neumann 边界条件
    for cx, cy, cz, r in spheres:
        # 判断输入点是否在球体表面
        on_sphere = torch.abs(torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - r) < 1e-3
        # 在球体表面的点使用球面的 Neumann 边界条件
        grad_u_n[on_sphere] = sphere_neumann_conditions(x[on_sphere], y[on_sphere], z[on_sphere], u[on_sphere], cx, cy, cz)
    
    return grad_u_n

def bc(num=100):
    x_bottom, y_bottom, z_bottom, _ = VPINN.rectangle_bc(x1, x2, y1, y2, z1, z2 / 2, lambda x, y, z: 0, num=num)
    x_top, y_top, z_top, _ = VPINN.rectangle_bc(x1, x2, y1, y2, z2 / 2, z2, lambda x, y, z: 0, num=num)
    xs = torch.cat([x_bottom, x_top], dim=0)
    ys = torch.cat([y_bottom, y_top], dim=0)
    zs = torch.cat([z_bottom, z_top], dim=0)
    operation = lambda x, y, z, u: neumann_conditions(x, y, z, u, x1, x2, y1, y2, z1, z2 / 2, z2)
    return [xs, ys, zs, torch.zeros_like(xs), operation]

# 初始化列表以存储所有的边界条件
bc_all = [bc(num=100)]

spheres = [(0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2), (0.2, 0.8, 0.7, 0.1), (0.6, 0.2, 0.3, 0.1)]
# 对每个球体，生成边界点并应用 Neumann 条件
for cx, cy, cz, r in spheres:
    operation = lambda x, y, z, u: neumann_conditions(x, y, z, u, x1, x2, y1, y2, z1, z2 / 2, z2)
    bc_sphere = sphere_bc(cx, cy, cz, r, num=100)
    bc_all.append(bc_sphere)

bc_all = [torch.cat([bc_all[0][i], bc_all[1][i]], dim=0) for i in range(4)]
bc_all.append(lambda x, y, z, u:combined_neumann_conditions(x, y, z, u, x1, x2, y1, y2, z1, z2 / 2, z2, spheres))

def pde(x, y, z, u):
    A=(20, 100)
    m=(1, 10, 5)
    k=(8, 10)
    mu=(1, 1)

    u_xx = VPINN.gradients(u, x, 2)
    u_yy = VPINN.gradients(u, y, 2)
    u_zz = VPINN.gradients(u, z, 2)

    def f(x, y, z):
        xlen2 = x**2 + y**2 + z**2
        part1 = torch.exp(torch.sin(m[0] * np.pi * x) + torch.sin(m[1] * np.pi * y) + torch.sin(m[2] * np.pi * z)) * (xlen2 - 1) / (xlen2 + 1)
        part2 = torch.sin(m[0] * np.pi * x) + torch.sin(m[1] * np.pi * y) + torch.sin(m[2] * np.pi * z)
        return A[0] * part1 + A[1] * part2

    interface_z = 0.5

    mus = torch.where(z < interface_z, mu[0], mu[1])
    ks = torch.where(z < interface_z, k[0]**2, k[1]**2)
    return -mus * (u_xx + u_yy + u_zz) + ks * u - f(x, y, z)

def plot_slices(u1, u2, x, y, z, direction='z', slices=10):
    # 转换为numpy数组
    u1 = u1.detach().numpy()
    u2 = u2.detach().numpy()
    x = x.detach().numpy()
    y = y.detach().numpy()
    z = z.detach().numpy()
    
    # 数据点
    points = np.hstack((x, y, z))
    
    # 建立网格
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    z_range = np.linspace(z_min, z_max, 100)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    
    # 对数据进行插值
    grid_u1 = griddata(points, u1[:, 0], (grid_x, grid_y, grid_z), method='linear')
    grid_u2 = griddata(points, u2[:, 0], (grid_x, grid_y, grid_z), method='linear')

    fig, axs = plt.subplots(nrows=slices, ncols=2, figsize=(10, slices * 5))

    if direction == 'x':
        coords = x_range
    elif direction == 'y':
        coords = y_range
    else: # 'z'
        coords = z_range

    indices = np.linspace(0, len(coords) - 1, num=slices, dtype=int)

    for ax, idx in zip(axs, indices):
        if direction == 'x':
            cax1 = ax[0].imshow(grid_u1[idx,:,:], extent=[y_min, y_max, z_min, z_max], origin='lower', aspect='auto')
            cax2 = ax[1].imshow(grid_u2[idx,:,:], extent=[y_min, y_max, z_min, z_max], origin='lower', aspect='auto')
        elif direction == 'y':
            cax1 = ax[0].imshow(grid_u1[:,idx,:], extent=[x_min, x_max, z_min, z_max], origin='lower', aspect='auto')
            cax2 = ax[1].imshow(grid_u2[:,idx,:], extent=[x_min, x_max, z_min, z_max], origin='lower', aspect='auto')
        else: # 'z'
            cax1 = ax[0].imshow(grid_u1[:,:,idx], extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
            cax2 = ax[1].imshow(grid_u2[:,:,idx], extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
        
        ax[0].set_title(f'{direction}={coords[idx]:.2f} (u1)')
        ax[1].set_title(f'{direction}={coords[idx]:.2f} (u2)')
        fig.colorbar(cax1, ax=ax[0])
        fig.colorbar(cax2, ax=ax[1])

    plt.tight_layout()
    plt.savefig('res.png')

def poisson3d_complex_geometry(epoch=10000):
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 20, 20, 20, 1],pde, bc2=bc(), area=[x1, x2, y1, y2, z1, z2], Q=10, grid_num=4, test_fcn_num=5, 
                device=device, load=None)

    net = vpinn.train('poisson3d_complex_geometry', epoch_num=epoch, coef=10)
    net.cpu()
    ################################################################################
    # verify
    data = np.loadtxt('poisson_3d.dat', skiprows=9)

    # get x、y、u of solution
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    u = data[:, 3]

    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    zz = torch.from_numpy(z).reshape(-1, 1).type(torch.float)
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

    net.cpu()

    prediction = net(torch.cat([xx, yy, zz], dim=1))
    solution = uu
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')

    # plot_slices(prediction, solution, xx, yy, zz, direction='x', slices=4)
    # fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    # axes = axs.flatten()
    
    # value = [u, prediction[:, 0].detach().numpy(), v, prediction[:, 1].detach().numpy(), p, prediction[:, 2].detach().numpy()]
    # tri = Triangulation(x, y)
    # name = ['u_real', 'u_pred', 'v_real', 'v_pred','p_real', 'p_pred']
    # for i in range(6):
    #     image = axes[i].tripcolor(tri, value[i], cmap='jet', edgecolors='k')
    #     axes[i].set_title(name[i])
    #     fig.colorbar(image, ax=axes[i])
    # plt.savefig('res.png')
    # return torch.norm(prediction - solution) / torch.norm(solution)
    return three_loss(prediction, solution)

if __name__ == "__main__":
    poisson3d_complex_geometry()