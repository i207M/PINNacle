import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
from .Integral import quad_integral2d
from .Integral import quad_integral3d
from .lengendre import test_func
from .net_class import MLP
from copy import deepcopy

class VPINN(ABC):
    @abstractmethod
    def loss_interior(self, net, device='cpu'):
        pass

    def train(self, model_name, epoch_num=20000, coef=[], task_id=0, queue=None, lr=1e-3, logevery=100,  plotevery=2000):
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # torch.autograd.set_detect_anomaly(True)
        
        net_need_plotting = []
        
        if self.load is not None:
            return ([self.net], self.net)

        with open(f'./log/task_id:{task_id}, {model_name}.txt', 'w') as file:
            pass
        
        for epoch in range(epoch_num):
            optimizer.zero_grad()
            # print(epoch)
            loss_names = ['loss_interior' ]
            losses = [self.loss_interior()]

            if self.constrains:
                for i in range(len(self.constrains)):
                    loss_names.append('loss_' + type(self.constrains[i]).__name__)
                    if self.constrains[i].inverse:
                        losses.append(coef[i] * self.constrains[i].loss(self.net_u, device=self.device))
                    else:
                        losses.append(coef[i] * self.constrains[i].loss(self.net, device=self.device))

            loss_tot = sum(losses)
            if epoch != 0 and (epoch + 1) % logevery == 0:
                loss_info = ''
                for i in range(len(losses)):
                    loss_info += f'{loss_names[i]}={losses[i].item():.5g}, '
                with open(f'./log/task_id:{task_id}, {model_name}.txt', 'a') as file:
                    file.write(f'epoch:{epoch + 1}'+'\n'+loss_info.rstrip(', ')+'\n')
            loss_tot.backward(retain_graph=True)
            optimizer.step()
            if queue:
                queue.put((task_id, epoch + 1))
        
            if model_name and epoch_num != 0 and (epoch + 1) % plotevery == 0:
                # print('\nhello\n')
                path = (f'./model/{model_name}{self.layer_sizes},Q={self.Q},grid_num={self.grid_num}'
                        f',test_fcn={self.test_fcn_num},coef={coef},epoch={epoch_num}).pth')
                torch.save(self.net, path)
                net_need_plotting.append(deepcopy(self.net))
        return (net_need_plotting, self.net)

class VPINN2d(VPINN):
    def __init__(self, layer_sizes, pde, constrains, grid_num, type='tanh', area = [-1, 1, -1, 1], Q=10, test_fcn_num=5, device='cpu', load=None, inverse=False, ref=None):
        self.pde = pde
        self.constrains = constrains
        self.Q = Q
        self.grid_num = grid_num
        self.test_fcn_num = test_fcn_num
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.layer_sizes = layer_sizes
        self.load = load
        self.inverse = inverse
        if inverse:
            self.u = ref
            self.net_u = MLP(layer_sizes, type).to(device)
        else:
            self.u = None
            
        if load:
            self.net = torch.load('./model/'+load).to(device)
        else:
            self.net = MLP(layer_sizes, type).to(device)
        
        # define the grid sample points
        quad_integral2d.init(Q, device)
        x = quad_integral2d.XX
        y = quad_integral2d.YY
        # (x1, y1) stands for the left down point of the rectangle
        # (x2, y2) stands for the right upper point of the rectangle
        x1 = area[0]
        x2 = area[1]
        y1 = area[2]
        y2 = area[3]
        lower_xs = torch.linspace(x1, x2, grid_num[0] + 1)[:-1]
        lower_ys = torch.linspace(y1, y2, grid_num[1] + 1)[:-1]
        xx, yy = torch.meshgrid(lower_xs, lower_ys, indexing='ij')
        x_bias = xx.reshape(-1, 1).to(device)
        y_bias = yy.reshape(-1, 1).to(device)
        
        regularized_x = (x.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_y = (y.reshape(1, -1).requires_grad_(True) + 1) / 2
        
        x_grid_len = (x2 - x1) / grid_num[0]
        y_grid_len = (y2 - y1) / grid_num[1]
        
        xs = regularized_x * x_grid_len + x_bias
        ys = regularized_y * y_grid_len + y_bias
        
        self.grid_xs = xs.reshape(-1, 1).to(device)
        self.grid_ys = ys.reshape(-1, 1).to(device)
        
        # define the test functions
        test_func.init(self.test_fcn_num)
        self.test_fcn0 = test_func.test_func(0, x, y)
        
        self.test_fcn0 = test_func.test_func(0, x, y).unsqueeze(-1).expand(-1, -1, -1, layer_sizes[-1])
        
        
    def lhsWrapper(self):
        X = torch.cat([self.grid_xs, self.grid_ys], dim=1)
        if self.inverse == False:
            U = self.net(X)
            lhs = self.pde(X, U)
            
        else:
            a = self.net(X)
            lhs = self.pde(X, self.net_u(X), a)
        
        result = torch.einsum('mcl,ncl->mncl', \
            lhs.view(self.grid_num[0] * self.grid_num[1], self.Q ** 2, self.layer_sizes[-1]), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2, self.layer_sizes[-1]))
        
        result = torch.reshape(result, (-1, self.Q ** 2, self.layer_sizes[-1]))
        return result
    
    def loss_interior(self):
        int1 = quad_integral2d.integral(self.lhsWrapper) * (1 / (self.grid_num[0] * self.grid_num[1]))
        int2 = torch.zeros_like(int1).requires_grad_(True)
        
        return self.loss(int1, int2)
    
        
class VPINN3d(VPINN):
    def __init__(self, layer_sizes, pde, constrains, grid_num, type='tanh', area = [-1, 1, -1, 1, -1, 1], Q=10, test_fcn_num=5, device='cpu', load=None, inverse=False, ref=None):
        self.pde = pde
        self.Q = Q
        self.grid_num = grid_num
        self.constrains = constrains
        self.test_fcn_num = test_fcn_num
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.layer_sizes = layer_sizes
        self.load = load
        self.inverse = inverse
        if inverse:
            self.u = ref
            self.net_u = MLP(layer_sizes, type).to(device)
        else:
            self.u = None

        if load:
            self.net = torch.load('./model/'+load).to(device)
        else:
            self.net = MLP(layer_sizes, type).to(device)
            
        # define the grid sample points
        quad_integral3d.init(Q, device)
        x = quad_integral3d.XX
        y = quad_integral3d.YY
        z = quad_integral3d.ZZ
        # (x1, y1) stands for the left down point of the rectangle
        # (x2, y2) stands for the right upper point of the rectangle
        x1 = area[0]
        x2 = area[1]
        y1 = area[2]
        y2 = area[3]
        z1 = area[4]
        z2 = area[5]
        lower_xs = torch.linspace(x1, x2, grid_num[0] + 1)[:-1]
        lower_ys = torch.linspace(y1, y2, grid_num[1] + 1)[:-1]
        lower_zs = torch.linspace(z1, z2, grid_num[2] + 1)[:-1]
        xx, yy, zz = torch.meshgrid(lower_xs, lower_ys, lower_zs,indexing='ij')
        x_bias = xx.reshape(-1, 1).to(device)
        y_bias = yy.reshape(-1, 1).to(device)
        z_bias = zz.reshape(-1, 1).to(device)
        
        regularized_x = (x.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_y = (y.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_z = (z.reshape(1, -1).requires_grad_(True) + 1) / 2
        
        x_grid_len = (x2 - x1) / grid_num[0]
        y_grid_len = (y2 - y1) / grid_num[1]
        z_grid_len = (z2 - z1) / grid_num[2]
        
        xs = regularized_x * x_grid_len + x_bias
        ys = regularized_y * y_grid_len + y_bias
        zs = regularized_z * z_grid_len + z_bias
        
        self.grid_xs = xs.reshape(-1, 1).to(device)
        self.grid_ys = ys.reshape(-1, 1).to(device)
        self.grid_zs = zs.reshape(-1, 1).to(device)
        
        # define the test functions
        test_func.init(self.test_fcn_num)
        self.test_fcn0 = test_func.test_func(0, x, y, z).unsqueeze(-1).expand(-1, -1, -1, layer_sizes[-1])
        # self.test_fcn1 = test_func.test_func(1, x, y, z)
        
        # check whether the laplace function is used
        self.pde = pde

    def lhsWrapper(self):
        X = torch.cat([self.grid_xs, self.grid_ys, self.grid_zs], dim=1)
        if self.inverse == False:
            u = self.net(X)
            lhs = self.pde(X, u)

        else:
            a = self.net(X)
            lhs = self.pde(X, self.net_u(X), a)

        result = torch.einsum('mcl,ncl->mncl', \
                lhs.view(self.grid_num[0] * self.grid_num[1] * self.grid_num[2], self.Q ** 3, self.layer_sizes[-1]), self.test_fcn0.view(self.test_fcn_num ** 3, self.Q ** 3, self.layer_sizes[-1]))
        
        result = result.reshape(-1, self.Q ** 3, self.layer_sizes[-1])
        return result
    
    def loss_interior(self):
        int1 = quad_integral3d.integral(self.lhsWrapper, self.layer_sizes[-1]) * (1 / (self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        int2 = torch.zeros_like(int1).requires_grad_(True)
        
        return self.loss(int1, int2)
    