import torch
import numpy as np
from abc import ABC, abstractmethod
from .grad import grad
from .geom import geom, rec, cube, circle, sphere, disk
from .geomtime import timeline, timeplane
from .net_class import MLP

class bc:
    # locate_func is used to specify some boundary points
    # all boundary points will be sampled if locate_func=None
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        self._device = value
        self.x = self.x.to(value)
        
    def __init__(self, geometry: geom, func, num, locate_func=None, u_component=0, inverse=False):
        self.geometry = geometry
        self.func = func
        self.locate_func = locate_func
        self.u_component = u_component
        self.num = num
        self.boundary_points = self.geometry.generate_boundary_points(self.num)
        self.tmp = None
        self._device = 'cpu'
        self.inverse = inverse
        
        # mask should be a tensor consisting of bools
        if (self.locate_func is not None) and ((self.__class__.__name__ == 'periodic') == False):
            mask = self.locate_func(self.boundary_points).squeeze()
            self.x = self.boundary_points[mask]
        else:
            self.x = self.boundary_points

    @abstractmethod
    def loss(self, net, device='cpu'):
        pass

class dirichlet(bc):
    
    def loss(self, net:MLP, device='cpu'):
        self.device = device
        U = net(self.x)
        if self.u_component == 'ALL':
            return torch.nn.MSELoss()(self.func(self.x), U)
        
        return torch.nn.MSELoss()(self.func(self.x), U[:, self.u_component:self.u_component + 1])

# the argument net is given in case of cube geometry
# only_side means we only care about side points of the geom 'disk'
def u_n_func(geom, u, x, locate_func, u_component, net=None, device='cpu', only_side=True):
    # this function returns points which value is identical to the given one in specific dimension
    def masked_x(component, value):
        locate_func = lambda x: torch.isclose(x[:, component], torch.full_like(x[:, component], value))
        return locate_func(x).squeeze()
        
    if isinstance(geom, timeline):
        u_n = grad(u, x, u_component=u_component, x_component=0, order=1)
        
    if isinstance(geom, timeplane) or isinstance(geom, rec):
        if locate_func is None:
            u_x = grad(u, x, u_component=u_component, x_component=0, order=1)
            u_y = grad(u, x, u_component=u_component, x_component=1, order=1)
            u_n = torch.cat([u_x[masked_x(0, geom.x1)], u_x[masked_x(0, geom.x2)], 
                        u_y[masked_x(1, geom.y1)], u_y[masked_x(1, geom.y2)]], dim=0)
            
        else:
            if (x[:, 0:1] <= geom.x1) | (x[:,0:1] >= geom.x2):
                u_n = grad(u, x, u_component=u_component, x_component=0, order=1)
        
            if (x[:, 1:2] <= geom.y1) | (x[:,1:2] >= geom.y2):
                u_n = grad(u, x, u_component=u_component, x_component=1, order=1)

    if isinstance(geom, cube):
        if locate_func is None:
            flg = [False, False, False]

            u_x = None
            u_y = None
            u_z = None

            if geom.left is not None: 
                x_ = [geom.left, geom.right] if geom.plane == False else [geom.left]
                bc_x = torch.cat(x_, dim=0).to(device)
                ux = net(bc_x)
                u_x = grad(ux, bc_x, u_component=u_component, x_component=0, order=1)
                flg[0] = True

            if geom.front is not None:
                y_ = [geom.front, geom.back] if geom.plane == False else [geom.front]
                bc_y = torch.cat(y_, dim=0).to(device)
                uy = net(bc_y)
                u_y = grad(uy, bc_y, u_component=u_component, x_component=1, order=1)
                flg[1] = True

            if geom.bottom is not None:
                z_ = [geom.bottom, geom.top] if geom.plane == False else [geom.bottom]
                bc_z = torch.cat(z_, dim=0).to(device)
                uz = net(bc_z)
                u_z = grad(uz, bc_z, u_component=u_component, x_component=2, order=1)
                flg[2] = True

            deriv = [u_x, u_y, u_z]

            u_n = torch.cat([ele for ele in deriv if ele is not None], dim=0)
    
        else:
            if (x[:, 0:1] <= geom.x1) | (x[:,0:1] >= geom.x2):
                u_n = grad(u, x, u_component=u_component, x_component=0, order=1)
        
            elif (x[:, 1:2] <= geom.y1) | (x[:,1:2] >= geom.y2):
                u_n = grad(u, x, u_component=u_component, x_component=1, order=1)
                
            elif (x[:, 2:3] <= geom.y1) | (x[:,2:3] >= geom.y2):
                u_n = grad(u, x, u_component=u_component, x_component=2, order=1)
    
    if isinstance(geom, circle):
        u_x = grad(u, x, u_component=u_component, x_component=0, order=1)
        u_y = grad(u, x, u_component=u_component, x_component=1, order=1)
        theta = torch.atan2(x[:, 1], x[:, 0])
        u_n = u_x * torch.cos(theta) + u_y * torch.sin(theta)
    
    if isinstance(geom, disk) and only_side == False:
        u_x = grad(u, x, u_component=u_component, x_component=0, order=1)
        u_y = grad(u, x, u_component=u_component, x_component=1, order=1)
        u_z = grad(u, x, u_component=u_component, x_component=2, order=1)
        
        # Bottom and top points
        mask_bottom = x[:, 2] == geom.b
        mask_top = x[:, 2] == geom.h
        u_n_bottom = -u_z[mask_bottom]
        u_n_top = u_z[mask_top]

        # Side points
        mask_side = torch.logical_not(torch.logical_or(mask_bottom, mask_top))
        x_side = x[mask_side]
        n_side = torch.stack([x_side[:, 0] - geom.x, x_side[:, 1] - geom.y, torch.zeros_like(x_side[:, 0])], dim=1)
        n_side /= torch.norm(n_side, dim=1, keepdim=True)  # Normalize the normal vectors
        u_n_side = u_x[mask_side] * n_side[:, 0:1] + u_y[mask_side] * n_side[:, 1:2]
        
        # Combine the results
        u_n = torch.zeros_like(u_z)
        u_n[mask_bottom] = u_n_bottom
        u_n[mask_top] = u_n_top
        u_n[mask_side] = u_n_side
    
    if isinstance(geom, disk) and only_side:
        u_x = grad(u, x, u_component=u_component, x_component=0, order=1)
        u_y = grad(u, x, u_component=u_component, x_component=1, order=1)
        
        # Only consider side points
        mask_side = torch.logical_and(x[:, 2] > geom.b, x[:, 2] < geom.h)
        x_side = x[mask_side]
        
        # normal vector
        n_side = torch.stack([x_side[:, 0] - geom.x, x_side[:, 1] - geom.y, torch.zeros_like(x_side[:, 0])], dim=1)
        n_side = n_side / torch.norm(n_side, dim=1, keepdim=True)  # Normalize the normal vectors
        u_n_side = u_x[mask_side] * n_side[:, 0:1] + u_y[mask_side] * n_side[:, 1:2]
        
        # Combine the results
        u_n = torch.zeros_like(u_x)
        u_n[mask_side] = u_n_side

    return u_n
    

class neumann(bc):

    def loss(self, net:MLP, device='cpu'):
        self.device = device
        u = net(self.x)
        u_n = u_n_func(self.geometry, u, self.x, self.locate_func, self.u_component, net, device)

        return torch.nn.MSELoss()(self.func(self.x), u_n)


class robin(bc):
    '''
     in case of robin boundary condition, the expected constrain is u_n = self.func(x, u)
    '''
    def loss(self, net:MLP, device='cpu'):
        self.device = device
        u = net(self.x)
        u_n = u_n_func(self.geometry, u, self.x, self.locate_func, self.u_component, net, device)
        
        return torch.nn.MSELoss()(self.func(self.x, u), u_n)
class periodic(bc):

    # in this case, the locate_func should be a list of 2 elements
    def loss(self, net:MLP, device='cpu'):
        boundary_points = self.geometry.generate_boundary_points(self.num).to(device)
        # points_chosen should be a tensor consisting of bools
        mask0 = self.locate_func[0](boundary_points).squeeze()
        mask1 = self.locate_func[1](boundary_points).squeeze()

        x0 = boundary_points[mask0]
        x1 = boundary_points[mask1]

        return torch.nn.MSELoss()(net(x0), net(x1))