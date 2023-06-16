import torch
import numpy as np
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

sin = torch.sin
cos = torch.cos
pi = torch.pi

def sample_points_on_circle(x, y, r, n):
    angles = torch.linspace(0, 2 * torch.pi, n)  # 在0到2π之间等间隔采样n个角度值
    x_coords = x + r * torch.cos(angles)  # 计算x坐标
    y_coords = y + r * torch.sin(angles)  # 计算y坐标

    return x_coords, y_coords


def f(x, y):
    return 10 * (17 + x ** 2 + y ** 2) * sin(pi * x) * sin(4 * pi * y)

def bc(boundary_num):
    xs = []
    ys = []
    us = []
    
    # sample on the circle
    circles = [(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, -0.7, 0.1), (-0.6, 0.5, 0.3)]
    for index in range(len(circles)):
        xx, yy = sample_points_on_circle(circles[index][0], circles[index][1], circles[index][2], boundary_num)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        xs.append(xx)
        ys.append(yy)
        us.append(torch.ones_like(xx))
    # circle_xs = torch.cat(xs, dim=0).view(-1, 1)
    # circle_ys = torch.cat(ys, dim=0).view(-1, 1)
    
    # sampel on the rectangle
    x1, y1, x2, y2 = (-1, -1, 1, 1)
    x_r = torch.linspace(x2, x2, boundary_num).reshape(-1, 1)
    y_r = torch.linspace(y1, y2, boundary_num).reshape(-1, 1)
            
    x_u = torch.linspace(x1, x2, boundary_num).reshape(-1, 1)
    y_u = torch.linspace(y2, y2, boundary_num).reshape(-1, 1)
                
    x_l = torch.linspace(x1, x1, boundary_num).reshape(-1, 1)
    y_l = torch.linspace(y1, y2, boundary_num).reshape(-1, 1)
                
    x_d = torch.linspace(x1, x2, boundary_num).reshape(-1, 1)
    y_d = torch.linspace(y1, y1, boundary_num).reshape(-1, 1)
                
    xs.extend([x_r, x_u, x_l, x_d])
    ys.extend([y_r, y_u, y_l, y_d])
    us.extend([torch.zeros_like(x_r), torch.zeros_like(x_u), torch.zeros_like(x_l), torch.zeros_like(x_d)])
    
    boundary_xs = torch.cat(xs, dim=0)
    boundary_ys = torch.cat(ys, dim=0)
    boundary_us = torch.cat(us, dim=0)
    return (boundary_xs, boundary_ys, boundary_us)

def in_circle(x, y):
    circles = [(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, -0.7, 0.1), (-0.6, 0.5, 0.3)]
    for i in range(len(circles)):
        if (x - circles[i][0]) ** 2 + (y - circles[i][1]) ** 2 < circles[i][2] ** 2:
            return True
    return False

# VPINN.laplace represents the result of laplace operator applied to u with Green Theorem
# When VPINN.laplace is used, you are expected to wrap the monomial with VPINN.LAPLACE_TERM() to distinguish
# VPINN.LAPLACE_TERM can only contain a monomial of VPINN.laplace, polynomial is not allowed

# def pde(x, y, u):    
#     return -VPINN.LAPLACE_TERM(VPINN.laplace(x, y, u)) + 64 * u - f(x, y)

# this pde doesn't use the green theorem to simplify the equation
def pde(x, y, u):    
    return -VPINN.gradients(u, x, 2) - VPINN.gradients(u, y, 2)  + 64 * u - f(x, y)

#############################################################################################
# train the model
def poisson_boltzmann2d(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([2, 10, 10, 10, 1],pde, bc1=bc(80), area=[-1, 1, -1, 1], Q=10, grid_num=16, test_fcn_num=5, 
                device=device, load=None)


    net = vpinn.train("Poisson_Boltzmann", epoch_num=epoch, coef=10 ** (3 - np.log2(vpinn.grid_num)))
    net.cpu()
    # verify
    data = np.loadtxt('poisson_boltzmann2d.dat', skiprows=9)

    # 提取x、y、u
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]

    x_tensor = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    y_tensor = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    u_tensor = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

    prediction_tensor = net(torch.cat([x_tensor, y_tensor], dim=1))
    print(f'median error={torch.median(torch.abs(u_tensor - prediction_tensor))}')
    print(f'relative error={torch.norm(u_tensor - prediction_tensor):.2f}/{torch.norm(u_tensor):.2f}={torch.norm(u_tensor - prediction_tensor) / torch.norm(u_tensor) * 100:.2f}%')

    # plot and verify
    xc = torch.linspace(-1, 1, 500)
    xx, yy = torch.meshgrid(xc, xc, indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)
    prediction = net(xy)

    # for i in range(500 ** 2):
    #     if in_circle(xx[i],yy[i]):
    #         # xx[i] = 
    #         # yy[i] = None
    #         prediction[i] = -1

    # prediction = prediction.reshape(500, 500)
    # prediction = prediction.transpose(0, 1)

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # fig.set_figwidth(14)
    # fig.set_figheight(6)
    # axes = ax.flatten()

    # # image 1
    # image1 = axes[0].imshow(prediction.detach().numpy(), cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
    # axes[0].set_title('Prediction')
    # fig.colorbar(image1, ax=axes[0])

    # # image 2
    # tri = Triangulation(x_tensor.reshape(-1), y_tensor.reshape(-1))
    # res = (u_tensor - prediction_tensor).to('cpu').reshape(-1).detach().numpy()
    # image2 = axes[1].tripcolor(tri, res, cmap='jet', edgecolors='k')
    # axes[1].set_title('Residual')
    # fig.colorbar(image2, ax=axes[1])
    # fig.tight_layout()
    # plt.savefig(f"error={torch.norm(u_tensor - prediction_tensor) / torch.norm(u_tensor) * 100 :.2f}%.png")
    # plt.show()
    # return torch.norm(u_tensor - prediction_tensor) / torch.norm(u_tensor)
    return three_loss(prediction_tensor, u_tensor)

if __name__ == "__main__":
    poisson_boltzmann2d()