import torch
import numpy as np
import copy
import sys, os
# import pyinstrument
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN2d import VPINN
from src.Loss import three_loss
script_path = os.path.abspath(__file__)

# 获取脚本所在的目录
script_dir = os.path.dirname(script_path)

# 切换到脚本所在的目录
os.chdir(script_dir)

bbox=[-10, 10, -10, 10]
split=(5, 5)
freq=2
# output dim
output_dim = 1
# geom
bbox = bbox

# PDE
a_cof = np.loadtxt("poisson_a_coef.dat")
f_cof = np.loadtxt("poisson_f_coef.dat").reshape(split[0], split[1], freq, freq)
block_size = np.array([(bbox[1] - bbox[0] + 2e-5) / split[0], (bbox[3] - bbox[2] + 2e-5) / split[1]])

def domain(x):
    reduced_x = (x - np.array(bbox[::2]) + 1e-5)
    dom = np.floor(reduced_x / block_size).astype("int32")
    return dom, reduced_x - dom * block_size

def a(x):
    dom, _ = domain(x)
    return a_cof[dom[0], dom[1]]

a = np.vectorize(a, signature="(2)->()")

def f(x):
    dom, res = domain(x)

    def f_fn(coef):
        ans = coef[0, 0]
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                tmp = np.sin(np.pi * np.array((i, j)) * (res / block_size))
                ans += coef[i, j] * tmp[0] * tmp[1]
        return ans

    return f_fn(f_cof[dom[0], dom[1]])

f = np.vectorize(f, signature="(2)->()")

def get_coef(x):
    x = x.detach().cpu()
    return torch.Tensor(a(x)).unsqueeze(dim=-1), torch.Tensor(f(x)).unsqueeze(dim=-1)

def pde(x, y, u, device='cuda'):
    u_xx = VPINN.gradients(u, x, 2)
    u_yy = VPINN.gradients(u, y, 2)
    
    a, f = get_coef(x)
    return a.to(device) * (u_xx + u_yy) + f.to(device)

def neumann_conditions(x, y, u, x1, x2, y1, y2):
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

def bc(num=100):
    frame = VPINN.rectangle_bc(-10, 10, -10, 10, lambda x, y:0, num=num)
    frame_x, frame_y = frame[0], frame[1]
    frame_u = torch.zeros_like(frame_x)
    operation = lambda x, y, u: neumann_conditions(x, y, u, -10, 10, -10, 10) + u
    return [frame_x, frame_y, frame_u, operation]


def Poisson2DManyArea(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([2, 20, 20, 20, 1], pde, 'tanh', bc2=bc(), area=[-10, 10, -10, 10],\
        Q=10, grid_num=4, test_fcn_num=5, device=device, load=None)
    
    net = vpinn.train('Poisson2DManyArea', epoch_num=epoch)
    net.cpu()
################################################################################
    # verify
    data = np.loadtxt('poisson_manyarea.dat', skiprows=9)

    # get x、y、u of solution
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]


    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

    net.cpu()

    prediction = net(torch.cat([xx, yy], dim=1))
    solution = uu
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axes = axs.flatten()
    
    # x_ = torch.from_numpy(x__).reshape(-1, 1).to(torch.float32)
    # y_ = torch.from_numpy(y__).reshape(-1, 1).to(torch.float32)
    # t_ = torch.from_numpy(t__).reshape(-1, 1).to(torch.float32)
    # pred = net(torch.cat([x_, y_, t_], dim=1)).detach().numpy()
    
    
    # value = [u, prediction[:, 0].detach().numpy()]
    # tri = Triangulation(x, y)
    # name = ['solution', 'prediction']
    # for i in range(2):
    #     image = axes[i].tripcolor(tri, value[i], cmap='jet', edgecolors='k')
    #     axes[i].set_title(name[i])
    #     fig.colorbar(image, ax=axes[i])
    # plt.savefig('res.png')
    
    # return torch.norm(prediction - solution) / torch.norm(solution)
    return three_loss(prediction, solution)

if __name__ == "__main__":
    Poisson2DManyArea()