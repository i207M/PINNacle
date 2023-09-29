import torch
from abc import ABC, abstractmethod
from .grad import grad
from .time import time
from .geomtime import timeline, timeplane
from .net_class import MLP
class ic:
    def __init__(self, domain: time, func, num, locate_func=None, u_component=0, inverse=False):
        '''
        Here domain is corresponding to the geom in bc.
        '''
        self.domain = domain
        self.func = func
        self.locate_func = locate_func
        self.u_component = u_component
        self.num = num
        self.inverse = inverse

    @abstractmethod
    def loss(self, net, device='cpu'):
        pass

class dirichlet(ic):

    def loss(self, net:MLP, device='cpu'):
        initial_points = self.domain.generate_initial_points(self.num).to(device)
        # points_chosen should be a tensor consisting of bools
        if self.locate_func is not None:
            mask = self.locate_func(initial_points)
            ic_points = initial_points[mask]
        else:
            ic_points = initial_points

        U = net(ic_points)
        if self.u_component == 'ALL':
            return torch.nn.MSELoss()(self.func(ic_points), U)
        
        return torch.nn.MSELoss()(self.func(ic_points)[:, self.u_component:self.u_component + 1], U[:, self.u_component:self.u_component + 1])

class neumann(ic):

    def loss(self, net, device='cpu'):
        initial_points = self.domain.generate_initial_points(self.num).to(device)
        # points_chosen should be a tensor consisting of bools
        if self.locate_func is not None:
            mask = self.locate_func(initial_points)
            ic_points = initial_points[mask]
        else:
            ic_points = initial_points

        U = net(ic_points)
        if isinstance(self.domain, timeline):
            u_t = grad(U, ic_points, u_component=self.u_component, x_component=1, order=1)
            
        if isinstance(self.domain, timeplane):
            u_t = grad(U, ic_points, u_component=self.u_component, x_component=2, order=1)
        
        return torch.nn.MSELoss()(self.func(ic_points), u_t)
    