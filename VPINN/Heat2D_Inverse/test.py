import torch
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
os.chdir(sys.path[0])
from src.VPINN3d import VPINN
from src.Loss import three_loss

pi = torch.pi
sin = torch.sin

def a_ref(x, y, t):  # irrelevent to time domain
    return 2 + sin(pi * x) * sin(pi * y)

def u_ref(x, y, t):
    return torch.exp(-t) * sin(pi * x) * sin(pi * y)

def f(x, y, t):
    s, c, p = torch.sin, torch.cos, pi
    
    return torch.exp(-t) * ((4*p**2-1) * s(p*x) * s(p*y) + \
        p**2 * (2 * s(p*x)**2 * s(p*y)**2 - c(p*x)**2 * s(p*y)**2 - s(p*x)**2 * c(p*y)**2))

def pde(x, y, t, u, a):
    u_t = VPINN.gradients(u, t)
    u_x = VPINN.gradients(u, x)
    u_y = VPINN.gradients(u, y)
    d_au = VPINN.gradients(a * u_x, x) + VPINN.gradients(a * u_y, y)
    
    return u_t - d_au - f(x, y, t)

def heat2d_inverse(epoch=10000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vpinn = VPINN([3, 20, 20, 20, 1], pde, 'tanh', VPINN.rectangle_bc(-1, 1, -1, 1, -1, 1, a_ref), area=[-1, 1, -1, 1, -1, 1],\
        Q=10, grid_num=8, test_fcn_num=5, device=device, load=None, inverse=True, ref=u_ref)
    
    net = vpinn.train('heat2d_inverse', epoch_num=epoch, coef=1)
    net.cpu()
    #############################################################################################
    # plot and verify
    N = 100
    xc = torch.linspace(-1, 1, N)
    xx, yy = torch.meshgrid(xc, xc, indexing='ij')
    xx = xx.reshape(-1, 1)
    t = 1
    yy = yy.reshape(-1, 1)
    xyt = torch.cat([xx, yy, torch.full_like(xx, t)], dim=1)
    prediction = net(xyt).reshape(N, N)
    prediction = prediction.transpose(0, 1)

    res = [None, None, None]
    for i in range(3):
        res[i] = prediction - a_ref(xx, yy, torch.full_like(xx, i * 0.5)).reshape(N, N)
        res[i] = res[i].reshape(N, N)
        res[i] = res[i].transpose(0, 1)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_figwidth(12)
    fig.set_figheight(10)
    axes = ax.flatten()

    image1 = axes[0].imshow((a_ref(xx, yy, torch.full_like(xx, 0)).reshape(N, N)).detach().numpy(), cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
    axes[0].set_title('Solution')
    fig.colorbar(image1, ax=axes[0])
    
    for i in range(1, 4):
        image = axes[i].imshow(res[i - 1].detach().numpy(), cmap='jet', origin='lower', extent=[-1, 1, -1, 1])
        axes[i].set_title(f'Residual, x={0.5 * (i - 1)}')
        fig.colorbar(image, ax=axes[i])
        fig.tight_layout()

    plt.savefig("res.png")
    print(f'relative error={(sum([torch.norm(res[i]) for i in range(3)]) / (3 * torch.norm(a_ref(xx, yy, torch.full_like(xx, 0))))).item() * 100:.2f}%')
    # plt.show()

    return three_loss(prediction.reshape(-1, 1), a_ref(xx, yy, torch.full_like(xx, 0).reshape(N, N)).requires_grad_(True))
    return (sum([torch.norm(res[i]) for i in range(3)]) / (3 * torch.norm(a_ref(xx, yy, torch.full_like(xx, 0)))))

if __name__ == "__main__":
    print(heat2d_inverse())