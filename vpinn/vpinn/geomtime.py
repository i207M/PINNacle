import torch
from abc import ABC, abstractmethod
from .geom import geom
from .time import time

class geom1dtime(geom, time):
    '''
    '''
    # @abstractmethod
    # def generate_boundary_points(self, num):
    #     pass

    # @abstractmethod
    # def generate_initial_points(self, num):
    #     pass

class timeline(geom1dtime):
    def __init__(self, x1, x2, t0, t, eps=2e-4):
        self.x1 = x1
        self.x2 = x2
        self.t = t
        self.eps = eps

    def generate_boundary_points(self, num):
        t = torch.cat([torch.linspace(0, self.t, num), 
                       torch.linspace(0, self.t, num)]).reshape(-1, 1)
        x = torch.cat([torch.linspace(self.x1, self.x1, num),
                       torch.linspace(self.x2, self.x2, num)]).reshape(-1,1)
        return torch.cat([x.requires_grad_(True), t.requires_grad_(True)], dim=1)
    
    def generate_initial_points(self, num):
        t = torch.linspace(0, 0, num).reshape(-1, 1)
        x = torch.linspace(self.x1 + self.eps, self.x2 - self.eps, num).reshape(-1, 1)
        return torch.cat([x.requires_grad_(True), t.requires_grad_(True)], dim=1)
    

class geom2dtime(geom, time):
    '''
    '''
    # @abstractmethod
    # def generate_boundary_points(self, num):
    #     pass

    # @abstractmethod
    # def generate_initial_points(self, num):
    #     pass

class timeplane(geom2dtime):
    """
    timeplane
    ======

    Provides
    1. A representation for a 2D plane over time.
    2. Inherits from the `geom2dtime` abstract base class.

    How to use
    ----------
    Instantiate a timeplane with the coordinates of the bottom left point, top right point, and time:

    >>> my_timeplane = timeplane(x1, x2, y1, y2, t)
    """
    def __init__(self, x1, x2, y1, y2, t0, t, eps=2e-4):
        """
        Initializes a timeplane object with given coordinates.

        Args:
        x1 (float): The x-coordinate of the bottom left point of the plane.
        x2 (float): The x-coordinate of the top right point of the plane.
        y1 (float): The y-coordinate of the bottom left point of the plane.
        y2 (float): The y-coordinate of the top right point of the plane.
        t0 (float): Always set to be 0
        t (float): The time at which the plane exists.
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.t = t
        self.eps = eps

    def generate_boundary_points(self, num):
        tt = torch.linspace(0 + self.eps, self.t - self.eps, num)
        xx = torch.linspace(self.x1 + self.eps, self.x2 - self.eps, num)
        yy = torch.linspace(self.y1 + self.eps, self.y2 - self.eps, num)
        
        t_x, x = torch.meshgrid(tt, xx, indexing='ij')
        t_y, y = torch.meshgrid(tt, yy, indexing='ij')

        t_x = t_x.reshape(-1,1)
        x = x.reshape(-1,1)
        t_y = t_y.reshape(-1,1)
        y = y.reshape(-1,1)

        t = torch.cat([t_x, t_x, t_y, t_y])
        x = torch.cat([x, x, torch.full((num**2, 1), self.x1), torch.full((num**2, 1), self.x2)])
        y = torch.cat([torch.full((num**2, 1), self.y1), torch.full((num**2, 1), self.y2), y, y])

        return torch.cat([x.requires_grad_(True), y.requires_grad_(True), t.requires_grad_(True)], dim=1)

    
    def generate_initial_points(self, num):
        t = torch.linspace(0, 0, num**2).reshape(-1, 1)
        x, y = torch.meshgrid(torch.linspace(self.x1, self.x2, num), torch.linspace(self.y1, self.y2, num), indexing='ij')
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        return torch.cat([x.requires_grad_(True), y.requires_grad_(True), t.requires_grad_(True)], dim=1)
