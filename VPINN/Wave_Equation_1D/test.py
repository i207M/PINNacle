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
def u(x, t):
    term1 = torch.sin(torch.pi*x)*torch.cos(2*torch.pi*t)
    term2 = 0.5*torch.sin(4*torch.pi*x)*torch.cos(8*torch.pi*t)
    result = term1 + term2
    return result


# VPINN.laplace represents the result of laplace operator applied to u with Green Theorem
# When VPINN.laplace is used, you are expected to wrap the monomial with VPINN.LAPLACE_TERM() to distinguish
# VPINN.LAPLACE_TERM can only contain a monomial of VPINN.laplace, polynomial is not allowed

def pde(x, t, u):    
    return VPINN.gradients(u, t, 2) - 4 * VPINN.gradients(u, x, 2)

# def pde2(x, t, u):    
    # return VPINN.gradients(u, t, 1)

# this pde doesn't use the green theorem to simplify the equation
# def pde(x, y, u):    
    # return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) - f(x, y)

#############################################################################################
# boundary condition
def bc(boundary_num):
    xs = []
    ys = []
    x1, y1, x2, y2 = (0, 0, 1, 1)
    x_r = torch.linspace(x2, x2, boundary_num).reshape(-1, 1)
    y_r = torch.linspace(y1, y2, boundary_num).reshape(-1, 1)
            
    # x_u = torch.linspace(x1, x2, boundary_num).reshape(-1, 1)
    # y_u = torch.linspace(y2, y2, boundary_num).reshape(-1, 1)
                
    x_l = torch.linspace(x1, x1, boundary_num).reshape(-1, 1)
    y_l = torch.linspace(y1, y2, boundary_num).reshape(-1, 1)
                
    x_d = torch.linspace(x1, x2, boundary_num).reshape(-1, 1)
    y_d = torch.linspace(y1, y1, boundary_num).reshape(-1, 1)
                
    xs.extend([x_r, x_l, x_d])
    ys.extend([y_r, y_l, y_d])
    boundary_xs = torch.cat(xs, dim=0)
    boundary_ys = torch.cat(ys, dim=0)
    boundary_us = u(boundary_xs, boundary_ys)
    return (boundary_xs, boundary_ys, boundary_us)

#############################################################################################
# train the model
def wave(epoch=10000):
    device = 'cuda'
    vpinn = VPINN([2, 10, 10, 10, 1], pde, bc1=bc(80), area=[0, 1, 0, 1], Q=10, grid_num=8, test_fcn_num=5, 
                device=device, load=None)


    net = vpinn.train("Wave", epoch_num=epoch, coef=1)

    #############################################################################################
    # plot and verify
    net.cpu()
    xc = torch.linspace(0, 1, 500)
    xx, yy = torch.meshgrid(xc, xc, indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)
    prediction = net(xy)
    res = prediction - u(xx, yy)
    prediction = torch.reshape(prediction, (500, 500))
    res = torch.reshape(res, (500, 500))
    solution = u(xx, yy).reshape(500, 500)

    prediction = prediction.transpose(0, 1)
    res = res.transpose(0, 1)
    solution = solution.transpose(0, 1)

    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # fig.set_figwidth(15)
    # fig.set_figheight(5)
    # axes = ax.flatten()

    # image1 = axes[0].imshow(prediction.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    # axes[0].set_title('Prediction')
    # fig.colorbar(image1, ax=axes[0])

    # image2 = axes[1].imshow(solution.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    # axes[1].set_title('solution')
    # fig.colorbar(image2, ax=axes[1])

    # image3 = axes[2].imshow(res.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    # axes[2].set_title('Residual')
    # fig.colorbar(image3, ax=axes[2])


    # fig.tight_layout()
    # plt.savefig("prediction_and_residual.png")
    print(f'relative error={(torch.norm(res) / torch.norm(u(xx, yy))).item() * 100:.2f}%')
    # plt.show()
    # return torch.norm(res) / torch.norm(u(xx, yy))
    return three_loss(prediction.reshape(-1, 1), u(xx, yy))

if __name__ == "__main__":
    wave()
