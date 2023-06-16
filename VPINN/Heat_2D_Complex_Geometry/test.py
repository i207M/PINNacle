import torch
import numpy as np
import sys, os
import copy
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

x1, x2, y1, y2, t1, t2 = (-8, 8, -12, 12, 0, 3)

def circle_normal_gradient(x, y, z, circle, u, net):
    x_c, y_c, _, _ = circle

    # 计算法线向量，并进行单位化
    normal = torch.cat([x - x_c, y - y_c], dim=1)
    normal = normal / torch.norm(normal)

    u = net(torch.cat([x, y, z], dim=1))
    # 计算u相对于x和y的偏导数
    du_x = VPINN.gradients(u, x, order=1)
    du_y = VPINN.gradients(u, y, order=1)

    # 计算法向导数
    u_n = normal[:,0:1] * du_x + normal[:,1:2] * du_y

    return u_n


def pde(x, y, t, u):
    return VPINN.gradients(u, t, 1) - (VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2))

def sample_points_on_cylinder(x, y, z, r, h, n=20, m=20):
    # 在0到2π之间等间隔采样n个角度值
    angles = torch.linspace(0, 2 * torch.pi, n)
    
    # 在0到h之间等间隔采样m个高度值
    heights = torch.linspace(0, h, m)

    # 使用torch.meshgrid在角度和高度上进行网格化采样
    angles, heights = torch.meshgrid(angles, heights, indexing='ij')

    # 计算x和y坐标
    x_coords = x + r * torch.cos(angles)
    y_coords = y + r * torch.sin(angles)

    # z坐标就是采样的高度值
    z_coords = z + heights

    # 使用torch.reshape将每个坐标的形状改为[n*m, 1]
    x_coords = torch.reshape(x_coords, [-1, 1])
    y_coords = torch.reshape(y_coords, [-1, 1])
    z_coords = torch.reshape(z_coords, [-1, 1])

    return x_coords, y_coords, z_coords


def bc(boundary_num=20):
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
    bc_xs = [xx, xt, xt, torch.full_like(xt, 0), torch.full_like(xt, 1)]
    bc_ys = [yy, torch.full_like(xt, 0), torch.full_like(xt, 1), xt, xt]
    bc_ts = [torch.zeros_like(xx), tt, tt, tt, tt]
    bc_us = [torch.sin(20 * torch.pi * xx) * torch.sin(torch.pi * yy), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0), torch.full_like(xt, 0)]
    
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_ts = torch.cat(bc_ts, dim=0).reshape(-1, 1)
    bc_us = torch.cat(bc_us, dim=0).reshape(-1, 1)
    
    return (bc_xs, bc_ys, bc_ts, bc_us)

def rec_neumann_conditions(x, y, z, u, x1, x2, y1, y2, z1, z2, net):
    u = net(torch.cat([x, y, z], dim=1))
    grad_u_x = VPINN.gradients(u, x)
    grad_u_y = VPINN.gradients(u, y)
    grad_u_z = VPINN.gradients(u, z)
    
    # 初始化结果张量
    grad_u_n = torch.zeros_like(x)
    
    # 对每个边界进行处理
    x_boundary = torch.where((x <= x1) | (x >= x2))
    y_boundary = torch.where((y <= y1) | (y >= y2))
    z_boundary = torch.where((z <= z1) | (z >= z2))

    # 根据位置设置边界条件
    grad_u_n[x_boundary] = grad_u_x[x_boundary]
    grad_u_n[y_boundary] = grad_u_y[y_boundary]
    grad_u_n[z_boundary] = grad_u_z[z_boundary]

    return grad_u_n

def batch_circle_normal_gradient(x, y, z, circles, u, len, net):

    # 初始化结果列表
    gradients = []

    # 对每一个圆进行操作
    for i, circle in enumerate(circles):
        # 提取对应圆的点和u值
        x_i = x[i*len:(i+1)*len]
        y_i = y[i*len:(i+1)*len]
        z_i = z[i*len:(i+1)*len]
        u_i = u[i*len:(i+1)*len]

        # 计算法向导数并存入结果列表
        gradient_i = circle_normal_gradient(x_i, y_i, z_i, circle, u_i, net)
        gradients.append(gradient_i)

    # 将结果列表转换为张量并返回
    return torch.cat(gradients)


def operation(xx, yy, zz, uu, net):
    circles = [(4, 3, 0, 1), (4, -3, 0, 1), (-4, 3, 0, 1), (-4, -3, 0, 1),
               (4, 9, 0, 1), (4, -9, 0, 1), (-4, 9, 0, 1), (-4, -9, 0, 1),
               (0, 0, 0, 1),
               (0, 6, 0, 1), (0, -6, 0, 1),
               (3.2, 6, 0, 0.4), (3.2, -6, 0, 0.4), (-3.2, 6, 0, 0.4), (-3.2, 6, 0, 0.4),
               (3.2, 0, 0, 0.4), (-3.2, 0, 0, 0.4)]
    
    delimeter1 = 11 * 20 ** 2
    delimeter2 = 17 * 20 ** 2
    g = [5, 1, 0.1]
    q = [1, 1, 1]

    x = [0, 0, 0]
    y = [0, 0, 0]
    z = [0, 0, 0]
    u = [0, 0, 0]

    grad = [0, 0, 0]

    x[0] = xx[:delimeter1]
    y[0] = yy[:delimeter1]
    z[0] = zz[:delimeter1]
    u[0] = uu[:delimeter1]

    x[1] = xx[delimeter1:delimeter2]
    y[1] = yy[delimeter1:delimeter2]
    z[1] = zz[delimeter1:delimeter2]
    u[1] = uu[delimeter1:delimeter2]

    x[2] = xx[delimeter2:]
    y[2] = yy[delimeter2:]
    z[2] = zz[delimeter2:]
    u[2] = uu[delimeter2:]

    
    for i in range(2):
        grad[i] = batch_circle_normal_gradient(x[i], y[i], z[i], circles, u[i], 20 ** 2, net)
    
    grad[2] = rec_neumann_conditions(x[2], y[2], z[2], uu, x1, x2, y1, y2, t1, t2, net)
    
    rhs = [0, 0, 0]
    for i in range(3):
        rhs[i] = g[i] - q[i] * u[i]
    
    grads = torch.cat(grad, dim=0)
    rhses = torch.cat(rhs, dim=0)
    return grads - rhses
    


def bc2(boundary_num=20):
    circles = [(4, 3, 0, 1), (4, -3, 0, 1), (-4, 3, 0, 1), (-4, -3, 0, 1),
               (4, 9, 0, 1), (4, -9, 0, 1), (-4, 9, 0, 1), (-4, -9, 0, 1),
               (0, 0, 0, 1),
               (0, 6, 0, 1), (0, -6, 0, 1),
               (3.2, 6, 0, 0.4), (3.2, -6, 0, 0.4), (-3.2, 6, 0, 0.4), (-3.2, 6, 0, 0.4),
               (3.2, 0, 0, 0.4), (-3.2, 0, 0, 0.4)]
    
    x = []
    y = []
    t = []

    for i in range(len(circles)):
        xx, yy, zz = sample_points_on_cylinder(circles[i][0], circles[i][1], circles[i][2], circles[i][3], t2)
        x.append(xx)
        y.append(yy)
        t.append(zz)
    
    x.append(bc()[0])
    y.append(bc()[1])
    t.append(bc()[2])

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    t = torch.cat(t, dim=0)
    u = torch.zeros_like(x)

    return x, y, t, u, operation

def heat_complex(epoch=10000):
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 30, 30, 30, 1],pde, type='tanh', bc2=bc2(100), area=[x1, x2, y1, y2, t1, t2], Q=10, grid_num=8, test_fcn_num=5, 
                device=device, load=None, inplace=True)

    # profiler=Profiler()
    # profiler.start()

    net = vpinn.train('heat_complex', epoch_num=epoch, coef=50)

    # profiler.stop()
    # profiler.print()
    # net = vpinn.train(None, epoch_num=0, coef=0.1)

    ################################################################################
    # verify
    net.cpu()
    data = np.loadtxt('heat_complex.dat', skiprows=9)

    # get x、y、u of solution
    x_ = data[:, 0]
    y_ = data[:, 1]
    x = []
    y = []
    t = []
    u = []
    for i in range(31):
        x.append(copy.deepcopy(x_))
        y.append(copy.deepcopy(y_))
        t.append([i * 0.1 for _ in range(len(x_))])
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

    prediction = net(torch.cat([xx, yy, tt], dim=1))
    # solution = u(xx, yy, zz)
    print(f'relative error={torch.norm(prediction - uu):.2f}/{torch.norm(uu):.2f}={torch.norm(prediction - uu) / torch.norm(uu) * 100:.2f}%')

    err = torch.norm(prediction - uu) / torch.norm(uu)
    # return err
    return three_loss(prediction, uu)
if __name__ == "__main__":
    heat_complex()
