import torch
import functools
import inspect
import regex as re
import matplotlib.pyplot as plt
from tqdm import tqdm
from .Integral import quad_integral3d
from .lengendre import test_func
from .net_class import MLP


class VPINN:
    def __init__(self, layer_sizes, pde, type='tanh', bc1=None, bc2=None, bc3=None, \
                 area = [-1, 1, -1, 1, -1, 1], Q=10, grid_num=4,test_fcn_num=5, device='cpu', load=None, inverse=False, ref=None, inplace=False):
        self.pde = pde
        self.bc1 = bc1
        self.bc2 = bc2
        self.bc3 = bc3
        self.Q = Q
        self.grid_num = grid_num
        self.test_fcn_num = test_fcn_num
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.layer_sizes = layer_sizes
        self.load = load
        self.inverse = inverse
        self.inplace = inplace
        if inverse:
            self.u = ref
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
        lower_xs = torch.linspace(x1, x2, grid_num + 1)[:-1]
        lower_ys = torch.linspace(y1, y2, grid_num + 1)[:-1]
        lower_zs = torch.linspace(z1, z2, grid_num + 1)[:-1]
        xx, yy, zz = torch.meshgrid(lower_xs, lower_ys, lower_zs,indexing='ij')
        x_bias = xx.reshape(-1, 1).to(device)
        y_bias = yy.reshape(-1, 1).to(device)
        z_bias = zz.reshape(-1, 1).to(device)
        
        regularized_x = (x.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_y = (y.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_z = (z.reshape(1, -1).requires_grad_(True) + 1) / 2
        
        x_grid_len = (x2 - x1) / grid_num
        y_grid_len = (y2 - y1) / grid_num
        z_grid_len = (z2 - z1) / grid_num
        
        xs = regularized_x * x_grid_len + x_bias
        ys = regularized_y * y_grid_len + y_bias
        zs = regularized_z * z_grid_len + z_bias
        
        self.grid_xs = xs.reshape(-1, 1).to(device)
        self.grid_ys = ys.reshape(-1, 1).to(device)
        self.grid_zs = zs.reshape(-1, 1).to(device)
        
        # pass the boundary sample pointf from arguments
        if bc1:
            self.bc1_xs = bc1[0].requires_grad_(True).to(device).reshape(-1,1)
            self.bc1_ys = bc1[1].requires_grad_(True).to(device).reshape(-1,1)
            self.bc1_zs = bc1[2].requires_grad_(True).to(device).reshape(-1,1)
            self.bc1_us = bc1[3].requires_grad_(True).to(device)
        # sometimes the bc1_us can be a tensor of multi dimensions. In this dimensions,
        # not all components are valid. This can happen in 2D_NS_lid_driven
            if len(bc1) > 4:
        # this should be a bool list, which consists of a tensor hasing same dimensions as 'bc1_us' has.
               self.bc1_validation = bc1[4]
            else:
                self.bc1_validation = None
        
        if bc2:
            self.bc2_xs = bc2[0].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_ys = bc2[1].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_zs = bc2[2].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_us = bc2[3].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_operation = bc2[4]
        
        if bc3:
            self.bc3_xs1 = bc3[0][0].requires_grad_(True).to(device).reshape(-1,1)
            self.bc3_ys1 = bc3[0][1].requires_grad_(True).to(device).reshape(-1,1)
            self.bc3_zs1 = bc3[0][2].requires_grad_(True).to(device).reshape(-1,1)
            self.bc3_xs2 = bc3[1][0].requires_grad_(True).to(device).reshape(-1,1)
            self.bc3_ys2 = bc3[1][1].requires_grad_(True).to(device).reshape(-1,1)
            self.bc3_zs2 = bc3[1][2].requires_grad_(True).to(device).reshape(-1,1)
            
        # define the test functions
        test_func.init(self.test_fcn_num)
        self.test_fcn0 = test_func.test_func(0, x, y, z).unsqueeze(-1).expand(-1, -1, -1, layer_sizes[-1])
        # self.test_fcn1 = test_func.test_func(1, x, y, z)
        
        # check whether the laplace function is used
        source_code = inspect.getsource(pde)
        
         
        laplace_term_pattern = r'\bLAPLACE_TERM\(((?:[^()]|\((?1)\))*)\)'
        
        laplace_term = re.search(laplace_term_pattern, source_code)
        calls_laplace = bool(laplace_term)
        self.calls_laplace = calls_laplace

        if calls_laplace:
            self.pde1 = self.__extract_laplace_term(pde, laplace_term.group(1).strip())
            self.pde = pde
        else:
            self.pde = pde
            self.pde1 = None

    def __extract_laplace_term(self, func, laplace_term):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return eval(laplace_term.replace('VPINN.laplace(x, y, u)', 'quad_integral3d.integral(self.Laplace)'), globals(), {'self': self})
        return wrapper

    # just serve as a placeholder
    @staticmethod
    def laplace(x, y, u):
        return torch.zeros_like(x)
    
    @staticmethod
    def LAPLACE_TERM(term):
        return torch.zeros_like(term)
    
    @staticmethod
    def gradients(u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True,
                                   allow_unused=True )[0]
        else:
            return VPINN.gradients(VPINN.gradients(u, x), x, order=order - 1)
    
    @staticmethod
    def rectangle_bc(x1, x2, y1, y2, z1, z2, func, num=100):
        xx = torch.linspace(x1, x2, num)
        yy = torch.linspace(y1, y2, num)
        zz = torch.linspace(z1, z2, num)
        xy_x, xy_y = torch.meshgrid(xx, yy, indexing='ij')
        xz_x, xz_z = torch.meshgrid(xx, zz, indexing='ij')
        yz_y, yz_z = torch.meshgrid(yy, zz, indexing='ij')

        xy_x = xy_x.reshape(-1,)
        xy_y = xy_y.reshape(-1,)
        xz_x = xz_z.reshape(-1,)
        xz_z = xz_z.reshape(-1,)
        yz_y = yz_z.reshape(-1,)
        yz_z = yz_z.reshape(-1,)

        x = [xy_x, xy_x, xz_x, xz_x, torch.linspace(x1, x1, num ** 2), torch.linspace(x2, x2, num ** 2)]
        y = [xy_y, xy_y, torch.linspace(y1, y1, num ** 2), torch.linspace(y2, y2, num ** 2), yz_y, yz_y]
        z = [torch.linspace(z1, z1, num ** 2), torch.linspace(z2, z2, num ** 2), xz_z, xz_z, yz_z, yz_z]
        x = torch.cat(x).reshape(-1, 1)
        y = torch.cat(y).reshape(-1, 1)
        z = torch.cat(z).reshape(-1, 1)
        u = func(x, y, z)
        return [x, y, z, u]
    
    @staticmethod
    def complex_cube_bc(x1, x2, y1, y2, z1, z2, num=100, func1=None, func2=None, func3=None, func4=None, func5=None, func6=None, valid=None):
        '''
        func1 对应立方体的右面，也就是 y=y2 的面。
        func2 对应立方体的左面，也就是 y=y1 的面。
        func3 对应立方体的前面，也就是 x=x2 的面。
        func4 对应立方体的后面，也就是 x=x1 的面。
        func5 对应立方体的顶面，也就是 z=z2 的面。
        func6 对应立方体的底面，也就是 z=z1 的面。
        '''
        funcs = [func1, func2, func3, func4, func5, func6]
        # Get the first function that is not None
        func_not_none = next((f for f in funcs if f is not None), None)
        
        # If all functions are None, set a default output size
        if func_not_none is None:
            output_size = 3
        else:
            # Get the output size of the function
            output_size = func_not_none(torch.tensor([0.0]).reshape(-1, 1), torch.tensor([0.0]).reshape(-1, 1), torch.tensor([0.0]).reshape(-1, 1)).numel()

        # Define default function if not provided
        default_func = lambda x, y, z: torch.zeros(x.size(0), output_size)
        for i in range(6):
            if funcs[i] is None:
                funcs[i] = default_func

        # Allocate space for the tensor
        n = num * num
        xs = torch.zeros(6 * n, 1)
        ys = torch.zeros(6 * n, 1)
        zs = torch.zeros(6 * n, 1)
        values = torch.zeros((6 * n, output_size))

        # If valid tensor is not provided, default to all True
        if valid is None:
            valid = [[True]*output_size]*6

        # Convert valid to a tensor
        valid = torch.tensor(valid, dtype=torch.bool)

        # Generate points on each edge
        coord1 = torch.linspace(x1, x2, num)
        coord2 = torch.linspace(y1, y2, num)
        coord3 = torch.linspace(z1, z2, num)

        for i in range(6):
            # For each face, create a grid of points
            if i < 2:
                grid1, grid2 = torch.meshgrid(coord1, coord3, indexing='ij')
            elif i < 4:
                grid1, grid2 = torch.meshgrid(coord2, coord3, indexing='ij')
            else:
                grid1, grid2 = torch.meshgrid(coord1, coord2, indexing='ij')
            grid1 = grid1.reshape(-1, 1)
            grid2 = grid2.reshape(-1, 1)

            if i < 2:
                xs[i*n:(i+1)*n, :] = grid1
                ys[i*n:(i+1)*n, :] = y2 if i == 0 else y1
                zs[i*n:(i+1)*n, :] = grid2
            elif i < 4:
                xs[i*n:(i+1)*n, :] = x2 if i == 2 else x1
                ys[i*n:(i+1)*n, :] = grid1
                zs[i*n:(i+1)*n, :] = grid2
            else:
                xs[i*n:(i+1)*n, :] = grid1
                ys[i*n:(i+1)*n, :] = grid2
                zs[i*n:(i+1)*n, :] = z2 if i == 4 else z1

            # Apply the function to the face
            values[i*n:(i+1)*n] = funcs[i](xs[i*n:(i+1)*n, :], ys[i*n:(i+1)*n, :], zs[i*n:(i+1)*n, :])

        validation = valid.repeat_interleave(n, dim=0)

        return [xs, ys, zs, values, validation]


    def Laplace(self):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        dx = VPINN.gradients(u, self.grid_xs, 1)
        dy = VPINN.gradients(u, self.grid_ys, 1)
        
        du = torch.cat([dx, dy], dim=1) * self.grid_num
        du = du.view(self.grid_num ** 2, self.Q ** 2, 2)
        dv = self.test_fcn1.view(self.test_fcn_num ** 2, self.Q ** 2, 2)
        
        du = du.unsqueeze(1).expand(-1, self.test_fcn_num ** 2, -1, -1)
        dv = dv.unsqueeze(0).expand(self.grid_num ** 2, -1, -1, -1)
        
        result = torch.sum(du * dv, dim=-1)
        result = result.view(-1, self.Q ** 2)
        return -result

    def lhsWrapper(self):
        if self.inverse == False:
            u = self.net(torch.cat([self.grid_xs, self.grid_ys, self.grid_zs], dim=1))
            lhs = self.pde(self.grid_xs, self.grid_ys, self.grid_zs, u)

        else:
            a = self.net(torch.cat([self.grid_xs, self.grid_ys, self.grid_zs], dim=1))
            lhs = self.pde(self.grid_xs, self.grid_ys, self.grid_zs, self.u(self.grid_xs, self.grid_ys, self.grid_zs), a)

        result = torch.einsum('mcl,ncl->mncl', \
                lhs.view(self.grid_num ** 3, self.Q ** 3, self.layer_sizes[-1]), self.test_fcn0.view(self.test_fcn_num ** 3, self.Q ** 3, self.layer_sizes[-1]))
        
        result = result.reshape(-1, self.Q ** 3, self.layer_sizes[-1])
        return result
    
    def lhsWrapper2(self):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys, self.grid_zs], dim=1))
        
        lhs = self.pde2(self.grid_xs, self.grid_ys, self.grid_zs, u)
        
        result = torch.einsum('mc,nc->mnc', \
            lhs.view(self.grid_num ** 3, self.Q ** 3), self.test_fcn0.view(self.test_fcn_num ** 3, self.Q ** 3))
        result = torch.reshape(result, (-1, self.Q ** 3))
        return result
    
    def loss_bc1(self):
        prediction = self.net(torch.cat([self.bc1_xs, self.bc1_ys, self.bc1_zs], dim=1))
        solution = self.bc1_us

        if self.bc1_validation != None:
            return self.loss(prediction[self.bc1_validation.view(prediction.shape)], solution[self.bc1_validation.view(solution.shape)])
        else:
            return self.loss(prediction, solution)
        
        # return self.loss(prediction, solution)
    
    def loss_bc2(self):
        u = self.net(torch.cat([self.bc2_xs, self.bc2_ys, self.bc2_zs], dim=1))
        if self.inplace == False:
            prediction = self.bc2_operation(self.bc2_xs, self.bc2_ys, self.bc2_zs,u)
        else:
            prediction = self.bc2_operation(self.bc2_xs, self.bc2_ys, self.bc2_zs, u, self.net)
        
        solution = self.bc2_us
        return self.loss(prediction, solution)
        # return torch.median(torch.abs(prediction - solution))
    
    def loss_bc3(self):
        echo1 = self.net(torch.cat([self.bc3_xs1, self.bc3_ys1, self.bc3_zs1], dim=1))
        echo2 = self.net(torch.cat([self.bc3_xs2, self.bc3_ys2, self.bc3_zs2], dim=1))
        return self.loss(echo1, echo2)
    
    def loss_interior(self):
        if self.calls_laplace == False:
            int1 = quad_integral3d.integral(self.lhsWrapper, self.layer_sizes[-1]) * ((1 / self.grid_num) ** 2)
        else:
            laplace_conponent = self.pde1(None, None, None) 
            rest = quad_integral3d.integral(self.lhsWrapper)
            int1 = (laplace_conponent + rest) * ((1 / self.grid_num) ** 2)
        
        int2 = torch.zeros_like(int1).requires_grad_(True)
        
        return self.loss(int1, int2)
        # return torch.median(torch.abs(int1 - int2))
    
    
    def train(self, model_name, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters())
            
        loss_validity = [True, self.bc1 is not None, self.bc2 is not None, self.bc3 is not None]
        loss_names = ['loss_interior', 'loss_dirichlet_bc', 'loss_neumann_bc', 'loss_periodic_bc']

        for epoch in tqdm(range(epoch_num)):
            optimizer.zero_grad()
            
            losses = [self.loss_interior()] + [coef*self.loss_bc1() if loss_validity[1] else 0,\
                    self.loss_bc2() if loss_validity[2] else 0, self.loss_bc3() if loss_validity[3] else 0]

            loss_tot = 0
            for i in range(4):
                if loss_validity[i]:
                    loss_tot += losses[i]

            if epoch % 1000 == 0:
                loss_info = ''
                for i in range(4):
                    if loss_validity[i]:
                        loss_info += f'{loss_names[i]}={losses[i].item():.5g}, '
                print(loss_info.rstrip(', '))        
            
            loss_tot.backward(retain_graph=True)
            optimizer.step()
        
        if model_name and epoch_num != 0:
            path = (f'./model/{model_name}{self.layer_sizes},Q={self.Q},grid_num={self.grid_num}'
                    f',test_fcn={self.test_fcn_num},epoch={epoch_num}).pth')
            torch.save(self.net, path)
        return self.net
        
