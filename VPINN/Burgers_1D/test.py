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

def f(x, t):
    return torch.zeros_like(x)

def bc(boundary_num):
    xs = []
    ts = []
    us = []
    x = torch.linspace(-1, 1, boundary_num).reshape(-1, 1)
    t = torch.zeros_like(x)
    u = -torch.sin(torch.pi * x)
    xs.append(x)
    ts.append(t)
    us.append(u)
    
    t = torch.linspace(0, 1, boundary_num).reshape(-1, 1)
    x1 = torch.full_like(t, -1)
    x2 = torch.full_like(t, 1)
    u = torch.zeros_like(t)
    
    xs.append(x1)
    ts.append(copy.deepcopy(t))
    us.append(copy.deepcopy(u))
    
    xs.append(x2)
    ts.append(copy.deepcopy(t))
    us.append(copy.deepcopy(u))
    
    boundary_xs = torch.cat(xs, dim=0)
    boundary_ts = torch.cat(ts, dim=0)
    boundary_us = torch.cat(us, dim=0)
    return (boundary_xs, boundary_ts, boundary_us)
    
def pde(x, t, u):
    return VPINN.gradients(u, t, 1) + u * VPINN.gradients(u, x, 1) - 0.01 / torch.pi * VPINN.gradients(u, x, 2)

def burgers_1d(epoch=10000):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # train the model
    vpinn = VPINN([2, 15, 15, 15, 1], pde, bc1=bc(100), area=[-1, 1, 0, 1], Q=10, grid_num=16, test_fcn_num=6, 
                device=device, load=None)

    # with pyinstrument.Profiler() as prof:
    net = vpinn.train("burgers1d", epoch_num=epoch, coef=0.001)
    # net = vpinn.train(None, epoch_num=0, coef=10)

    # print(prof.output_text(unicode=True, color=True))


    # verify
    data = np.loadtxt('burgers1d.dat', skiprows=8)

    # get x、y、u of solution
    x_ = data[:, 0]
    x = []
    t = []
    u = []
    for i in range(11):
        x.append(copy.deepcopy(x_))
        t.append([i * 0.1 for _ in range(len(x_))])
        u.append(data[:, i + 1])

    x = np.concatenate(x)
    t = np.concatenate(t)
    u = np.concatenate(u)
    tri = Triangulation(x, t)

    x_tensor = torch.from_numpy(x).reshape(-1, 1).type(torch.float).to(device)
    y_tensor = torch.from_numpy(t).reshape(-1, 1).type(torch.float).to(device)
    u_tensor = torch.from_numpy(u).reshape(-1, 1).type(torch.float).to(device)

    verify_tensor = net(torch.cat([x_tensor, y_tensor], dim=1))
    print(f'median error={torch.median(torch.abs(u_tensor - verify_tensor))}')
    print(f'relative error={torch.norm(u_tensor - verify_tensor) / torch.norm(u_tensor) * 100:.2f}%')

    # plot and verify
    xc = torch.linspace(-1, 1, 500)
    yc = torch.linspace(0, 1, 500)
    xx, yy = torch.meshgrid(xc, yc, indexing='ij')
    xx = xx.reshape(-1, 1).to(device)
    yy = yy.reshape(-1, 1).to(device)
    xy = torch.cat([xx, yy], dim=1)
    prediction = net(xy).to('cpu')
    xx = xx.to('cpu')
    yy = yy.to('cpu')
    prediction = prediction.reshape(500, 500)
    prediction = prediction.transpose(0, 1)


    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(14)
    fig.set_figheight(5)
    axes = ax.flatten()

    # image 1
    image1 = axes[0].imshow(prediction.detach().numpy(), cmap='jet', origin='lower', extent=[-1, 1, 0, 1])
    axes[0].set_title('Prediction')
    fig.colorbar(image1, ax=axes[0])

    # image 2
    tri = Triangulation(x, t)
    res = (u_tensor - verify_tensor).to('cpu').reshape(-1).detach().numpy()
    image2 = axes[1].tripcolor(tri, res, cmap='jet', edgecolors='k')
    axes[1].set_title('Residual')
    fig.colorbar(image2, ax=axes[1])
    fig.tight_layout()
    plt.savefig(f"error={torch.norm(u_tensor - verify_tensor) / torch.norm(u_tensor) * 100:.2f}%.png")
    # plt.show()
    return three_loss(verify_tensor, u_tensor)
    return torch.norm(u_tensor - verify_tensor) / torch.norm(u_tensor)

if __name__ == "__main__":
    burgers_1d()