import torch
import numpy as np
import sys, os
# import pyinstrument
import matplotlib.pyplot as plt
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN2d import VPINN
from src.Loss import three_loss

#############################################################################################
# define the pde
def u(x, y):
    return (0.1 * torch.sin(2 * torch.pi * x) + torch.tanh(10 * x)) * torch.sin(2 * torch.pi * y)

def f(x, y, m=1, n=1, c=0.1, k=10):
    term1 = (4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x) +
             (2 * k**2 * torch.sinh(k * x) / torch.cosh(k * x)**3)) * torch.sin(2 * torch.pi * n * y)
    term2 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x)) * torch.sin(2 * torch.pi * n * y)
    return -(term1 + term2)

# VPINN.laplace represents the result of laplace operator applied to u with Green Theorem
# When VPINN.laplace is used, you are expected to wrap the monomial with VPINN.LAPLACE_TERM() to distinguish
# VPINN.LAPLACE_TERM can only contain a monomial of VPINN.laplace, polynomial is not allowed

def pde(x, y, u):    
    return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) - f(x, y)

# this pde doesn't use the green theorem to simplify the equation
# def pde(x, y, u):    
    # return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) - f(x, y)

#############################################################################################
# boundary condition
def bc(boundary_num):
    xs = []
    ys = []
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
    boundary_xs = torch.cat(xs, dim=0)
    boundary_ys = torch.cat(ys, dim=0)
    boundary_us = u(boundary_xs, boundary_ys)
    return (boundary_xs, boundary_ys, boundary_us)

#############################################################################################
# train the model
def poisson2d(epoch=10000):
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([2, 10, 10, 10, 1],pde, type='tanh',bc1=bc(80), area=[-1, 1, -1, 1], Q=30, grid_num=6, test_fcn_num=5, 
                device=device, load=None)


    net = vpinn.train("Poisson_1", epoch_num=epoch, coef=1)
    net.cpu()
    #############################################################################################
    # plot and verify
    xc = torch.linspace(-1, 1, 500)
    xx, yy = torch.meshgrid(xc, xc, indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)
    prediction = net(xy)
    res = prediction - u(xx, yy)
    prediction = torch.reshape(prediction, (500, 500))
    res = torch.reshape(res, (500, 500))

    prediction = prediction.transpose(0, 1)
    res = res.transpose(0, 1)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(13)
    fig.set_figheight(5)
    axes = ax.flatten()

    image1 = axes[0].imshow(prediction.detach().numpy(), cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
    axes[0].set_title('Prediction')
    fig.colorbar(image1, ax=axes[0])

    image2 = axes[1].imshow(res.detach().numpy(), cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
    axes[1].set_title('Residual')
    fig.colorbar(image2, ax=axes[1])
    fig.tight_layout()
    plt.savefig(f"error={torch.norm(res) / torch.norm(u(xx, yy)) * 100 :.2f}%.png")
    print(f'relative error={(torch.norm(res) / torch.norm(u(xx, yy))).item() * 100:.2f}%')
    # plt.show()
    # return torch.norm(res) / torch.norm(u(xx, yy))
    return three_loss(prediction, u(xx, yy))

if __name__ == "__main__":
    poisson2d()