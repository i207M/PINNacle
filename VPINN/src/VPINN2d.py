import torch
import functools
import inspect
import regex as re
from tqdm import tqdm
from .Integral import quad_integral2d
from .lengendre import test_func
from .net_class import MLP

class VPINN:
    def __init__(self, layer_sizes, pde, type='tanh', bc1=None, bc2=None, bc3=None,area = [-1, 1, -1, 1], Q=10, grid_num=4, test_fcn_num=5, device='cpu', load=None, inverse=False, ref=None):
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
        if inverse:
            self.u = ref
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
        lower_xs = torch.linspace(x1, x2, grid_num + 1)[:-1]
        lower_ys = torch.linspace(y1, y2, grid_num + 1)[:-1]
        xx, yy = torch.meshgrid(lower_xs, lower_ys, indexing='ij')
        x_bias = xx.reshape(-1, 1).to(device)
        y_bias = yy.reshape(-1, 1).to(device)
        
        regularized_x = (x.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_y = (y.reshape(1, -1).requires_grad_(True) + 1) / 2
        
        x_grid_len = (x2 - x1) / grid_num
        y_grid_len = (y2 - y1) / grid_num
        
        xs = regularized_x * x_grid_len + x_bias
        ys = regularized_y * y_grid_len + y_bias
        
        self.grid_xs = xs.reshape(-1, 1).to(device)
        self.grid_ys = ys.reshape(-1, 1).to(device)
        
        # pass the boundary sample point from arguments
        if bc1:
            self.bc1_xs = bc1[0].requires_grad_(True).to(device).reshape(-1,1)
            self.bc1_ys = bc1[1].requires_grad_(True).to(device).reshape(-1,1)
            self.bc1_us = bc1[2].requires_grad_(True).to(device).reshape(-1,1)
        # sometimes the bc1_us can be a tensor of multi dimensions. In this dimensions,
        # not all components are valid. This can happen in 2D_NS_lid_driven
            if len(bc1) > 3:
        # this should be a bool list, which consists of a tensor hasing same dimensions as 'bc1_us' has.
               self.bc1_validation = bc1[3]
            else:
                self.bc1_validation = None
        
        if bc2:
            self.bc2_xs = bc2[0].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_ys = bc2[1].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_us = bc2[2].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_operation = bc2[3]
    
        # define the test functions
        test_func.init(self.test_fcn_num)
        self.test_fcn0 = test_func.test_func(0, x, y)
        self.test_fcn0 = test_func.test_func(0, x, y).unsqueeze(-1).expand(-1, -1, -1, layer_sizes[-1])
        self.test_fcn1 = test_func.test_func(1, x, y)
        
        # check whether the laplace function is used
        source_code = inspect.getsource(pde)
        
         
        laplace_term_pattern = r'\bLAPLACE_TERM\(((?:[^()]|\((?1)\))*)\)'
        
        laplace_term = re.search(laplace_term_pattern, source_code)
        calls_laplace = bool(laplace_term)
        self.calls_laplace = calls_laplace

        if calls_laplace:
            self.pde1 = self.extract_laplace_term(pde, laplace_term.group(1).strip())
            self.pde = pde
        else:
            self.pde = pde
            self.pde1 = None

    def extract_laplace_term(self, func, laplace_term):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return eval(laplace_term.replace('VPINN.laplace(x, y, u)', 'quad_integral2d.integral(self.Laplace)'), globals(), {'self': self})
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
                                   only_inputs=True, )[0]
        else:
            return VPINN.gradients(VPINN.gradients(u, x), x, order=order - 1)
    
    @staticmethod
    def rectangle_bc(x1, x2, y1, y2, func, num=100):
        x = [torch.linspace(x1, x2, num), torch.linspace(x1, x2, num), torch.linspace(x1, x1, num), torch.linspace(x2, x2, num)]
        y = [torch.linspace(y1, y1, num), torch.linspace(y2, y2, num), torch.linspace(y1, y2, num), torch.linspace(y1, y2, num)]
        x = torch.cat(x).reshape(-1, 1)
        y = torch.cat(y).reshape(-1, 1)
        u = func(x, y)
        return [x, y, u]
    
    @staticmethod
    def complex_rec_bc(x1, x2, y1, y2, num=100, func1=None, func2=None, func3=None, func4=None, valid=None):
        funcs = [func1, func2, func3, func4]
        '''
        The order is top, right, bottom, left
        '''
        # Get the first function that is not None
        func_not_none = next((f for f in funcs if f is not None), None)
        
        # If all functions are None, set a default output size
        if func_not_none is None:
            output_size = 3
        else:
            # Get the output size of the function
            output_size = func_not_none(torch.tensor([0.0]).reshape(-1, 1), torch.tensor([0.0]).reshape(-1, 1)).numel()

        # Define default function if not provided
        default_func = lambda x, y: torch.zeros(output_size)
        if func1 is None:
            func1 = default_func
        if func2 is None:
            func2 = default_func
        if func3 is None:
            func3 = default_func
        if func4 is None:
            func4 = default_func

        # Allocate space for the tensor
        xs = torch.zeros(4*num, 1)
        ys = torch.zeros(4*num, 1)
        values = torch.zeros((4*num, output_size))

        # If valid tensor is not provided, default to all True
        if valid is None:
            valid = [[True]*output_size]*4

        # Prepare the validation list
        validation = []
        
        # Generate points on each edge
        top_edge_x = torch.linspace(x1, x2, num).unsqueeze(1)
        right_edge_y = torch.linspace(y1, y2, num).unsqueeze(1)
        bottom_edge_x = top_edge_x.clone()
        left_edge_y = right_edge_y.clone()

        # Store x, y coordinates and apply the functions to each edge
        xs[0*num:1*num, :] = top_edge_x
        ys[0*num:1*num, :] = torch.full_like(top_edge_x, y2)       # Fill tensor with y2
        values[0*num:1*num] = func1(top_edge_x, ys[0*num:1*num])         # Top edge value
        validation.extend(valid[0]*num)

        xs[1*num:2*num, :] = torch.full_like(right_edge_y, x2)    # Fill tensor with x2
        ys[1*num:2*num, :] = right_edge_y
        values[1*num:2*num] = func2(xs[1*num:2*num], right_edge_y)       # Right edge value
        validation.extend(valid[1]*num)

        xs[2*num:3*num, :] = bottom_edge_x
        ys[2*num:3*num, :] = torch.full_like(bottom_edge_x, y1)   # Fill tensor with y1
        values[2*num:3*num] = func3(bottom_edge_x, ys[2*num:3*num])      # Bottom edge value
        validation.extend(valid[2]*num)

        xs[3*num:4*num, :] = torch.full_like(left_edge_y, x1)     # Fill tensor with x1
        ys[3*num:4*num, :] = left_edge_y
        values[3*num:4*num] = func4(xs[3*num:4*num], left_edge_y)        # Left edge value
        validation.extend(valid[3]*num)

        validation = torch.tensor(validation, dtype=torch.bool).reshape(4*num, output_size)

        return [xs, ys, values, validation]


    def Laplace(self, x_=None, y_=None, u_=None):
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

    def lhsWrapper(self, x=None, y=None, u_in=None):
        if self.inverse == False:
            u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
            lhs = self.pde(self.grid_xs, self.grid_ys, u)
            
        else:
            a = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
            lhs = self.pde(self.grid_xs, self.grid_ys, self.u(self.grid_xs, self.grid_ys), a)
        
        result = torch.einsum('mcl,ncl->mncl', \
            lhs.view(self.grid_num ** 2, self.Q ** 2, self.layer_sizes[-1]), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2, self.layer_sizes[-1]))
        
        # result = torch.einsum('mc,nc->mnc', \
            # lhs.view(self.grid_num ** 2, self.Q ** 2), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2))
        result = torch.reshape(result, (-1, self.Q ** 2, self.layer_sizes[-1]))
        # result = torch.reshape(result, (-1, self.Q ** 2))
        return result
    
    def loss_bc1(self):
        prediction = self.net(torch.cat([self.bc1_xs, self.bc1_ys], dim=1))
        solution = self.bc1_us

        if self.bc1_validation != None:
            return self.loss(prediction[self.bc1_validation.view(prediction.shape)], solution[self.bc1_validation.view(solution.shape)])
        else:
            return self.loss(prediction, solution)
        
    def loss_bc2(self):
        u = self.net(torch.cat([self.bc2_xs, self.bc2_ys], dim=1))
        prediction = self.bc2_operation(self.bc2_xs, self.bc2_ys, u)
        solution = self.bc2_us
        return self.loss(prediction, solution)
    
    def loss_interior(self):
        if self.calls_laplace == False:
            int1 = quad_integral2d.integral(self.lhsWrapper) * ((1 / self.grid_num) ** 2)
        else:
            laplace_conponent = self.pde1(None, None, None) 
            rest = quad_integral2d.integral(self.lhsWrapper)
            int1 = (laplace_conponent + rest) * ((1 / self.grid_num) ** 2)
        
        int2 = torch.zeros_like(int1).requires_grad_(True)
        
        return self.loss(int1, int2)
    
    def train(self, model_name, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters())
            
        # for i in tqdm(range(epoch_num)):
        #     optimizer.zero_grad()
        #     if self.bc2:
        #         loss = self.loss_interior() + coef * (self.loss_bc1() + self.loss_bc2())
        #     else:
        #         loss = self.loss_interior() + coef * self.loss_bc1()
            
        #     if i % 1000 == 0:
        #         print(f'loss_interior={self.loss_interior().item():.5g}, loss_bc={self.loss_bc1().item():.5g}, coef={coef}')
        #     loss.backward(retain_graph=True)
        #     optimizer.step()
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
        
