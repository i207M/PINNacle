import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def u(x, t):
    term1 = torch.sin(torch.tensor(torch.pi*x))*torch.cos(torch.tensor(2*torch.pi*t))
    term2 = 0.5*torch.sin(torch.tensor(4*torch.pi*x))*torch.cos(torch.tensor(8*torch.pi*t))
    result = term1 + term2
    return result

def bc(boundary_num):
    xs = []
    ys = []
    x1, y1, x2, y2 = (0, 0, 1, 1)
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

x, y, u = bc(100)

# tri = Triangulation(boundary_xs.reshape(-1), boundary_ys.reshape(-1))
# res = (boundary_us).to('cpu').reshape(-1).detach().numpy()
# plt.tripcolor(tri, res, cmap='jet', edgecolors='k')
plt.scatter(x, y, c=u, cmap='coolwarm')
plt.colorbar()
plt.tight_layout()
# plt.savefig("prediction_and_residual.png")
plt.show()