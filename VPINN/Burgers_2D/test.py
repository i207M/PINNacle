import torch
import numpy as np
import sys, os
import copy
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from matplotlib.tri import Triangulation
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

L = 4
def pde(x, y, t, u):
    u1 = u[:,0:1]
    u2 = u[:,1:2]
    u1_t = VPINN.gradients(u1, t, 1)
    u2_t = VPINN.gradients(u2, t, 1)
    
    u1_x = VPINN.gradients(u1, x, 1)
    u2_x = VPINN.gradients(u2, x, 1)
    
    u1_y = VPINN.gradients(u1, y, 1)
    u2_y = VPINN.gradients(u2, y, 1)
    
    u1_xx = VPINN.gradients(u1, x, 2)
    u1_yy = VPINN.gradients(u1, y, 2)
    
    u2_xx = VPINN.gradients(u2, x, 2)
    u2_yy = VPINN.gradients(u2, y, 2)
    v = 0.001
    
    return torch.cat([u1_t + u1 * u1_x + u2 * u1_y - v * (u1_xx + u1_yy),
                     u2_t + u1 * u2_x + u2 * u2_y - v * (u2_xx + u2_yy)], dim=1)

def bc(boundary_num=10):
    ic_coefs = np.loadtxt("burgers2d_coef.dat")
    
    def ic_func(x, component):
        A = ic_coefs[:2*(2*L+1)**2].reshape(2, 2*L+1, 2*L+1)
        B = ic_coefs[2*(2*L+1)**2: 4*(2*L+1)**2].reshape(2, 2*L+1, 2*L+1)
        C = ic_coefs[4*(2*L+1)**2:]

        w = np.zeros((x.shape[0], 1))
        for i in range(-L, L + 1):
            for j in range(-L, L + 1):
                w += A[component][i][j] * np.sin(2 * np.pi * (i * x[:, 0:1] + j * x[:, 1:2])) \
                    + B[component][i][j] * np.cos(2 * np.pi * (i * x[:, 0:1] + j * x[:, 1:2]))

        return 2 * w + C[component]  # Note: change "divide w.max" to "devide M", where M is a constant param
    
    x = torch.linspace(0, L, boundary_num)
    y = torch.linspace(0, L, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    tt = torch.zeros_like(xx)
    x = torch.cat([xx, yy, tt], dim=1)
    u1 = torch.from_numpy(ic_func(x.numpy(), 0))
    u2 = torch.from_numpy(ic_func(x.numpy(), 1))
    u = torch.cat([u1, u2], dim=1)
    return [xx, yy, tt, u.to(torch.float32)]

def bc3(boundary_num=50):
    x = torch.linspace(0, L, boundary_num)
    y = torch.linspace(0, L, boundary_num)
    t = torch.linspace(0, 1, boundary_num)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xt, tt = torch.meshgrid(x, t, indexing='ij')
    
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xt = xt.reshape(-1, 1)
    tt = tt.reshape(-1, 1)
    
    xs1 = torch.cat([xt, torch.full_like(xt, 0)], dim=0)
    ys1 = torch.cat([torch.full_like(xt, 0), xt], dim=0)
    zs1 = torch.cat([tt, tt], dim=0)
    
    xs2 = torch.cat([xt, torch.full_like(xt, L)], dim=0)
    ys2 = torch.cat([torch.full_like(xt, L), xt], dim=0)
    zs2 = torch.cat([tt, tt], dim=0)
    #########################################################################
    bc3 = [[xs1, ys1, zs1], [xs2, ys2, zs2]]
    return bc3
    
def burgers2d(epoch=10000):
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 30, 30, 30, 2],pde, type='relu', bc1=bc(100), bc3=bc3(), area=[0, 4, 0, 4, 0, 1], Q=10, grid_num=6, test_fcn_num=5, 
                device=device, load='Burgers2d[3, 30, 30, 30, 2],Q=10,grid_num=6,test_fcn=5,epoch=20000).pth')

    # profiler=Profiler()
    # profiler.start()

    net = vpinn.train('Burgers2d', epoch_num=epoch, coef=10)

    # profiler.stop()
    # profiler.print()
    # net = vpinn.train(None, epoch_num=0, coef=0.1)

    ################################################################################
    # verify
    data = np.loadtxt('burgers2d.dat', skiprows=9)

    # get x、y、u of solution
    x_ = data[:, 0]
    y_ = data[:, 1]
    data_len = len(x_)
    x = []
    y = []
    t = []
    u = []
    v = []
    for i in range(11):
        x.append(copy.deepcopy(x_))
        y.append(copy.deepcopy(y_))
        t.append([i * 0.1 for _ in range(len(x_))])
        u.append(data[:, 2 * i + 2])
        v.append(data[:, 2 * i + 3])

    x = np.concatenate(x)
    y = np.concatenate(y)
    t = np.concatenate(t)
    u = np.concatenate(u)
    v = np.concatenate(v)
    # tri = Triangulation(x, t)

    xx = torch.from_numpy(x).reshape(-1, 1).type(torch.float)
    yy = torch.from_numpy(y).reshape(-1, 1).type(torch.float)
    tt = torch.from_numpy(t).reshape(-1, 1).type(torch.float)
    uu = torch.from_numpy(u).reshape(-1, 1).type(torch.float)
    vv = torch.from_numpy(v).reshape(-1, 1).type(torch.float)

    # x_initial = xx[0:3000]
    # y_initial = yy[0:3000]
    # t_initial = tt[0:3000]
    # u_initial = uu[0:3000]
    net.cpu()
    # sol = torch.sin(20 * torch.pi * x_initial) * torch.sin(torch.pi * y_initial)
    # pred = net(torch.cat([x_initial, y_initial, t_initial], dim=1))
    # diff = pred - u_initial
    # x = torch.linspace(-1, 1, 50)
    # y = x
    # z = x
    # xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    # xx = xx.reshape(-1, 1)
    # yy = yy.reshape(-1, 1)
    # zz = zz.reshape(-1, 1)

    prediction = net(torch.cat([xx, yy, tt], dim=1))
    solution = torch.cat([uu, vv], dim=1)
    print(f'relative error={torch.norm(prediction - solution):.2f}/{torch.norm(solution):.2f}={torch.norm(prediction - solution) / torch.norm(solution) * 100:.2f}%')
    

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axes = axs.flatten()
    image = [None, None, None, None]
    
    x__ = x[0: data_len]
    y__ = y[0: data_len]
    t__ = t[0: data_len]
    
    x_ = torch.from_numpy(x__).reshape(-1, 1).to(torch.float32)
    y_ = torch.from_numpy(y__).reshape(-1, 1).to(torch.float32)
    t_ = torch.from_numpy(t__).reshape(-1, 1).to(torch.float32)
    pred = net(torch.cat([x_, y_, t_], dim=1)).detach().numpy()
    
    u__ = u[0: data_len]
    v__ = v[0: data_len]
    
    value = [u__, pred[:, 0], v__, pred[:, 1]]
    tri = Triangulation(x__, y__)
    name = ['u_real', 'u_pred', 'v_real', 'v_pred']
    for i in range(4):
        image[i] = axes[i].tripcolor(tri, value[i], cmap='jet', edgecolors='k')
        axes[i].set_title(name[i])
        fig.colorbar(image[i], ax=axes[i])
    plt.savefig('res.png')
    # err = torch.norm(prediction - uu) / torch.norm(uu)
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # axes = axs.flatten()
    # for i in range(6):
    #     # z = torch.tensor(-1 + 0.5 * i)
    #     # zz = torch.full_like(xx, -1 + i * 0.5)
    #     # zz = zz.reshape(-1, 1)

    #     xx = xx[0:3000]
    #     yy = yy[0:3000]
    #     prediction = net(torch.cat([xx, yy, tt[3000 * (5 * i): 3000 * (5 * i + 1)]], dim=1)).detach()
    #     solution = uu[3000 * (5 * i): 3000 * (5 * i + 1)]
    #     res = (prediction - solution).reshape(-1).detach().numpy()
            
    #     tri = Triangulation(xx.reshape(-1), yy.reshape(-1))

    #     image = axes[i].tripcolor(tri, res, cmap='jet', edgecolors='k')
    #     axes[i].set_title(f'z={i}')
    #     fig.colorbar(image, ax=axes[i])
    # plt.savefig('res.png')
    torch.cuda.empty_cache()
    return three_loss(prediction, solution)
    # return err

if __name__ == "__main__":
    burgers2d()
