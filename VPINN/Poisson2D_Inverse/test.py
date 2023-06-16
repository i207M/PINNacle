import torch
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN2d import VPINN
from src.Loss import three_loss

pi = torch.pi
sin = torch.sin

def a_ref(x, y):
    return 1 / (1 + x ** 2 + y ** 2 + (x - 1) ** 2 + (y - 1) ** 2)

def u_ref(x, y):
    return sin(pi * x) * sin(pi * y)
        
def f(x, y):
    return 2 * pi**2 * torch.sin(pi * x) * torch.sin(pi * y) * a_ref(x, y) + \
        2 * pi * ((2*x+1) * torch.cos(pi * x) * torch.sin(pi * y) + (2*y+1) * torch.sin(pi * x) * torch.cos(pi * y)) * a_ref(x, y)**2

def pde(x, y, u, a):
    u_x = VPINN.gradients(u, x)
    u_y = VPINN.gradients(u, y)
    d_au = VPINN.gradients(a * u_x, x) + VPINN.gradients(a * u_y, y)
    
    return d_au + f(x, y)

def poisson_inverse(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([2, 20, 20, 20, 1], pde, 'tanh', VPINN.rectangle_bc(0, 1, 0, 1, a_ref), area=[0, 1, 0, 1],\
        Q=10, grid_num=4, test_fcn_num=5, device=device, load=None, inverse=True, ref=u_ref)
    
    net = vpinn.train('poisson_inverse', epoch_num=epoch, coef=10)
    net.cpu()
    #############################################################################################
    # plot and verify
    N = 100
    xc = torch.linspace(0, 1, N)
    xx, yy = torch.meshgrid(xc, xc, indexing='ij')
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)
    xy = torch.cat([xx, yy], dim=1)
    prediction = net(xy)
    res = prediction - a_ref(xx, yy)
    prediction = prediction.reshape(N, N)
    res = res.reshape(N, N)

    prediction = prediction.transpose(0, 1)
    res = res.transpose(0, 1)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(12)
    fig.set_figheight(5)
    axes = ax.flatten()

    image1 = axes[0].imshow(prediction.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    axes[0].set_title('Prediction')
    fig.colorbar(image1, ax=axes[0])

    image2 = axes[1].imshow(res.detach().numpy(), cmap='jet', origin='lower', extent=[0, 1, 0, 1])
    axes[1].set_title('Residual')
    fig.colorbar(image2, ax=axes[1])
    fig.tight_layout()
    plt.savefig("res.png")
    print(f'relative error={(torch.norm(res) / torch.norm(a_ref(xx, yy))).item() * 100:.2f}%')
    # plt.show()
    # return torch.norm(res) / torch.norm(a_ref(xx, yy))
    return three_loss(prediction.reshape(-1, 1), a_ref(xx, yy))

if __name__ == "__main__":
    poisson_inverse()

