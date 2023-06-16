import torch
import numpy as np
import sys, os
import torch.nn as nn
from pyinstrument import Profiler
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

def u(x, y, z):
    return x ** 2 + y ** 2 + z ** 2 + 10

def pde(x, y, z, u):
    return VPINN.gradients(u, x, 2) + VPINN.gradients(u, y, 2) + VPINN.gradients(u, z, 2) - 6

def bc(boundary_num=10):
    x = torch.linspace(-1, 1, boundary_num)
    y = torch.linspace(-1, 1, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    bc_xs = [xx, xx, xx, xx, torch.full_like(xx, 1), torch.full_like(yy, 1)]
    bc_ys = [yy, yy, torch.full_like(xx, 1), torch.full_like(xx, -1), xx, xx]
    bc_zs = [torch.full_like(xx, 1), torch.full_like(xx, -1), yy, yy, yy, yy]
    bc_xs = torch.cat(bc_xs, dim=0).reshape(-1, 1)
    bc_ys = torch.cat(bc_ys, dim=0).reshape(-1, 1)
    bc_zs = torch.cat(bc_zs, dim=0).reshape(-1, 1)
    bc_us = u(bc_xs, bc_ys, bc_zs)
    return (bc_xs, bc_ys, bc_zs, bc_us)

def poisson3d(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 20, 20, 20, 1],pde, bc1=bc(10), area=[-1, 1, -1, 1, -1, 1], Q=10, grid_num=4, test_fcn_num=5, 
                device=device, load=None)

    net = vpinn.train('Poisson3d', epoch_num=epoch, coef=10)

    ################################################################################
    net.cpu()
    x = torch.linspace(-1, 1, 50)
    y = x
    z = x
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    zz = zz.reshape(-1, 1)

    prediction = net(torch.cat([xx, yy, zz], dim=1))
    solution = u(xx, yy, zz)
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axes = axs.flatten()
    x = torch.linspace(-1, 1, 100)
    y = x
    for i in range(4):
        z = torch.tensor(-1 + 0.5 * i)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        zz = torch.full_like(xx, -1 + i * 0.5)
        zz = zz.reshape(-1, 1)

        prediction = net(torch.cat([xx, yy, zz], dim=1))
        solution = u(xx, yy, zz)
        res = (prediction - solution).reshape(100, 100)
        image = axes[i].imshow(res.detach().numpy(), cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
        axes[i].set_title(f'z={-1 + 0.5 * i}')
        fig.colorbar(image, ax=axes[i])
    plt.savefig('res.png')
    # return torch.norm(prediction - solution) / torch.norm(solution)
    return three_loss(prediction, solution)

if __name__ == "__main__":
    poisson3d()
