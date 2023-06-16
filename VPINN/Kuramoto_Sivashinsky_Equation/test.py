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


def bc(boundary_num):
    x = torch.linspace(0, torch.pi, boundary_num).reshape(-1, 1)
    t = torch.zeros_like(x)
    u = torch.cos(x) * (1 + torch.sin(x))
    
    return [x, t, u]
    
def pde(x, t, u):
    alpha =  100 / 16
    beta = 100 / (16 ** 2)
    gamma = 100 / (16 ** 4)
    u_t = VPINN.gradients(u, t, 1)
    u_x = VPINN.gradients(u, x, 1)
    u_xx = VPINN.gradients(u, x, 2)
    u_xxxx = VPINN.gradients(u, x, 4)
    return u_t + alpha *u * u_x + beta * u_xx + gamma * u_xxxx

x1, x2, t1, t2 = (0, torch.pi, 0, 1)

def kuramoto(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # train the model
    vpinn = VPINN([2, 15, 15, 15, 1], pde, bc1=bc(100), area=[x1, x2, t1, t2], Q=10, grid_num=16, test_fcn_num=6, 
                device=device, load=None)

    # with pyinstrument.Profiler() as prof:
    net = vpinn.train("kuramoto", epoch_num=epoch, coef=1)
    # verify
    data = np.loadtxt('Kuramoto_Sivashinsky.dat', skiprows=0)
    net.cpu()
    # get x、y、u of solution
    x = data[:, 0]
    t = data[:, 1]
    u = data[:, 2]

    tri = Triangulation(x, t)

    x_tensor = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    y_tensor = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
    u_tensor = torch.from_numpy(u).reshape(-1, 1).type(torch.float)

    xy = torch.cat([x_tensor, y_tensor], dim=1)
    prediction = net(xy)
    print(f'median error={torch.median(torch.abs(u_tensor - prediction))}')
    print(f'relative error={torch.norm(u_tensor - prediction) / torch.norm(u_tensor) * 100:.2f}%')

    # # plot and verify
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    # axes = ax.flatten()
    # # image 1
    # image1 = axes[0].tripcolor(tri, u_tensor.reshape(-1).detach().numpy(), cmap='jet', edgecolors='k')
    # axes[0].set_title('Solution')
    # fig.colorbar(image1, ax=axes[0])

    # # image 2
    # res = u_tensor - prediction
    # image2 = axes[1].tripcolor(tri, prediction.reshape(-1).detach().numpy(), cmap='jet', edgecolors='k')
    # axes[1].set_title('prediction')
    # fig.colorbar(image2, ax=axes[1])
    # fig.tight_layout()
    # plt.savefig('res.png')
    # plt.show()
    # return torch.norm(u_tensor - prediction) / torch.norm(u_tensor)
    return three_loss(prediction, u_tensor)

if __name__ == "__main__":
    kuramoto()