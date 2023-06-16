import torch
from .GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights

# os.chdir(sys.path[0])
class Quad_Integral2d:
    
    def init(self, Q, device='cpu'):
        self.Q = Q
        a, b = 0, 0
        [X, W] = GaussLobattoJacobiWeights(Q, a, b)

        X, Wx = torch.tensor(X,dtype=torch.float32), torch.tensor(W, dtype=torch.float32)
        Y, Wy = X, Wx
        self.XX, self.YY = torch.meshgrid(X, Y, indexing='ij')
        self.Wxx, self.Wyy = torch.meshgrid(Wx, Wy, indexing='ij')
        self.XX = self.XX.reshape(-1, 1).to(device)
        self.YY = self.YY.reshape(-1, 1).to(device)
        self.Wxx = self.Wxx.reshape(-1, 1).to(device)
        self.Wyy = self.Wyy.reshape(-1, 1).to(device)
    
    def integral(self, func, l=1):
        integral = torch.sum(func() * (self.Wxx * self.Wyy).squeeze(-1).unsqueeze(0).unsqueeze(-1).expand(-1, -1, l), dim=1)
        # integral = torch.sum(func() * (self.Wxx * self.Wyy).squeeze(-1).unsqueeze(0).expand(-1, self.Q ** 2), dim=1)
        return integral

quad_integral2d = Quad_Integral2d()

class Quad_Integral3d:
    
    def init(self, Q, device='cpu'):
        self.Q = Q
        a, b = 0, 0
        [X, W] = GaussLobattoJacobiWeights(Q, a, b)

        X, Wx = torch.tensor(X,dtype=torch.float32), torch.tensor(W, dtype=torch.float32)
        Y, Wy = X, Wx
        Z, Wz = X, Wx
        self.XX, self.YY, self.ZZ = torch.meshgrid(X, Y, Z, indexing='ij')
        self.Wxx, self.Wyy, self.Wzz = torch.meshgrid(Wx, Wy, Wz, indexing='ij')
        self.XX = self.XX.reshape(-1, 1).to(device)
        self.YY = self.YY.reshape(-1, 1).to(device)
        self.ZZ = self.ZZ.reshape(-1, 1).to(device)
        self.Wxx = self.Wxx.reshape(-1, 1).to(device)
        self.Wyy = self.Wyy.reshape(-1, 1).to(device)
        self.Wzz = self.Wzz.reshape(-1, 1).to(device)
    
    def integral(self, func, l=1):
        integral = torch.sum(func() * (self.Wxx * self.Wyy * self.Wzz).squeeze(-1).unsqueeze(0).unsqueeze(-1).expand(-1, -1, l), dim=1)
        return integral
    
quad_integral3d = Quad_Integral3d()
# quad_integral3d.init(10)
# xx, yy, zz = (quad_integral3d.XX, quad_integral3d.YY, quad_integral3d.ZZ)
# # xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
# xx = xx.reshape(1, -1)
# yy = yy.reshape(1, -1)
# zz = zz.reshape(1, -1)

# def f(x=xx, y=yy, z=zz):
#     return torch.sin(x ** 2) * torch.cos(y ** 2) * torch.tan(z ** 2)

# print(quad_integral3d.integral(f))