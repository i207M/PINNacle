import torch
from abc import ABC, abstractmethod

class geom(ABC):
    @abstractmethod
    def generate_boundary_points(self, num):
        pass

class geom2d(geom):
    """
    geom2d
    ======

    Provides
    1. An abstract class for representing 2D geometries.
    2. An abstract method `method` that should be implemented in child classes.
    """
    # @abstractmethod
    # def generate_boundary_points(self, num):
        # pass
        
class line(geom2d):
    def __init__(self, point1, point2):
        """
        Initializes a line object with given coordinates.

        Args:
        point1 (tuple): The (x, y) coordinates of the first point of the line.
        point2 (tuple): The (x, y) coordinates of the second point of the line.
        """
        self.x1, self.y1 = point1
        self.x2, self.y2 = point2
        
    def generate_boundary_points(self, num=100):
        """
        Generates boundary points on the line.

        Args:
        num (int): The number of points to generate. Defaults to 100.

        Returns:
        torch.Tensor: A tensor of size (num, 2) representing the (x, y) coordinates of the boundary points.
        """
        # Linearly interpolate between the two points
        x = torch.linspace(self.x1, self.x2, num).reshape(-1, 1)
        y = torch.linspace(self.y1, self.y2, num).reshape(-1, 1)
        
        # Return the concatenated tensor
        return torch.cat([x.requires_grad_(True), y.requires_grad_(True)], dim=1)
        
class rec(geom2d):
    """
    rec (rectangle)
    ===

    Provides
    1. A representation for 2D rectangle geometry.
    2. Inherits from the `geom2d` abstract base class.

    How to use
    ----------
    Instantiate a rectangle with the coordinates of the bottom left and top right points:

    >>> rectangle = rec(x1, x2, y1, y2)
    """
    def __init__(self, x1, x2, y1, y2, eps=2e-4):
        """
        Initializes a rectangle object with given coordinates.

        Args:
        x1 (float): The x-coordinate of the bottom left point of the rectangle.
        x2 (float): The x-coordinate of the top right point of the rectangle.
        y1 (float): The y-coordinate of the bottom left point of the rectangle.
        y2 (float): The y-coordinate of the top right point of the rectangle.
        num (int): The number of points to generate for each edge. Defaults to 100.
        eps (float): Small offset used for the points generation. Defaults to 2e-4.
        """
        
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
        self.xx1 = x1 + eps
        self.xx2 = x2 - eps
        self.yy1 = y1 + eps
        self.yy2 = y2 - eps

        self.num = 0

    def generate_boundary_points(self, num=100):
        if self.num != num:
            # define the four edges of the rectangle
            self.left = torch.cat([torch.ones(num).reshape(-1, 1)*self.x1, torch.linspace(self.yy1, self.yy2, num).reshape(-1, 1)], dim=1).requires_grad_(True)
            self.right = torch.cat([torch.ones(num).reshape(-1, 1)*self.x2, torch.linspace(self.yy1, self.yy2, num).reshape(-1, 1)], dim=1).requires_grad_(True)
            self.bottom = torch.cat([torch.linspace(self.xx1, self.xx2, num).reshape(-1, 1), torch.ones(num).reshape(-1, 1)*self.y1], dim=1).requires_grad_(True)
            self.top = torch.cat([torch.linspace(self.xx1, self.xx2, num).reshape(-1, 1), torch.ones(num).reshape(-1, 1)*self.y2], dim=1).requires_grad_(True)
        return torch.cat([self.left, self.right, self.bottom, self.top], dim=0)
    
class circle(geom2d):
    """
    circle
    ======

    Provides
    1. A representation for 2D circle geometry.
    2. Inherits from the `geom2d` abstract base class.

    How to use
    ----------
    Instantiate a circle with the coordinates of the center and radius:

    >>> my_circle = circle(x, y, r)
    """
    def __init__(self, x, y, r):
        """
        Initializes a circle object with given center coordinates and radius.

        Args:
        x (float): The x-coordinate of the center of the circle.
        y (float): The y-coordinate of the center of the circle.
        r (float): The radius of the circle.
        """
        self.x = x
        self.y = y
        self.r = r

    def generate_boundary_points(self, num=100):
        theta = torch.linspace(0, 2 * torch.pi, num).reshape(-1, 1)  # angle
        # convert polar coordinates to cartesian coordinates
        x = self.x + self.r * torch.cos(theta)
        y = self.y + self.r * torch.sin(theta)
        return torch.cat([x.requires_grad_(True), y.requires_grad_(True)], dim=1)
    
class geom3d(geom):
    """
    geom3d
    ======

    Provides
    1. An abstract class for representing 3D geometries.
    2. An abstract method `method` that should be implemented in child classes.
    """
    @abstractmethod
    def generate_boundary_points(self, num):
        pass

class cube(geom3d):
    """
    cube
    ====

    Provides
    1. A representation for 3D cube geometry.
    2. Inherits from the `geom3d` abstract base class.

    How to use
    ----------
    Instantiate a cube with the coordinates of the bottom left corner and the top right corner:

    >>> my_cube = cube(x1, x2, y1, y2, z1, z2)
    """
    def __init__(self, x1, x2, y1, y2, z1, z2, eps=2e-4):
        """
        Initializes a cube object with given coordinates.

        Args:
        x1 (float): The x-coordinate of the bottom left corner of the cube.
        x2 (float): The x-coordinate of the top right corner of the cube.
        y1 (float): The y-coordinate of the bottom left corner of the cube.
        y2 (float): The y-coordinate of the top right corner of the cube.
        z1 (float): The z-coordinate of the bottom left corner of the cube.
        z2 (float): The z-coordinate of the top right corner of the cube.
        num (int): The number of points to generate for each edge. Defaults to 100.
        eps (float): Small offset used for the points generation. Defaults to 2e-4.
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.z1 = z1
        self.z2 = z2
        
        self.xx1 = x1 + eps
        self.xx2 = x2 - eps
        self.yy1 = y1 + eps
        self.yy2 = y2 - eps
        self.zz1 = z1 + eps
        self.zz2 = z2 - eps

        self.num = 0
        self.plane = False
    def generate_boundary_points(self, num):
        if self.num != num:
            # generate points for each face
            xx = torch.linspace(self.xx1, self.xx2, num)
            yy = torch.linspace(self.yy1, self.yy2, num)
            zz = torch.linspace(self.zz1, self.zz2, num)
            xy_x, xy_y = torch.meshgrid(xx, yy, indexing='ij')
            xz_x, xz_z = torch.meshgrid(xx, zz, indexing='ij')
            yz_y, yz_z = torch.meshgrid(yy, zz, indexing='ij')

            self.bottom = torch.cat([xy_x.reshape(-1, 1), xy_y.reshape(-1, 1), torch.full((num**2, 1), self.z1)], dim=1).requires_grad_(True) if (self.x1 != self.x2 and self.y1 != self.y2) else None
            self.top = torch.cat([xy_x.reshape(-1, 1), xy_y.reshape(-1, 1), torch.full((num**2, 1), self.z2)], dim=1).requires_grad_(True) if (self.x1 != self.x2 and self.y1 != self.y2) else None
            self.front = torch.cat([xz_x.reshape(-1, 1), torch.full((num**2, 1), self.y1), xz_z.reshape(-1, 1)], dim=1).requires_grad_(True) if (self.x1 != self.x2 and self.z1 != self.z2) else None
            self.back = torch.cat([xz_x.reshape(-1, 1), torch.full((num**2, 1), self.y2), xz_z.reshape(-1, 1)], dim=1).requires_grad_(True) if (self.x1 != self.x2 and self.z1 != self.z2) else None
            self.left = torch.cat([torch.full((num**2, 1), self.x1), yz_y.reshape(-1, 1), yz_z.reshape(-1, 1)], dim=1).requires_grad_(True) if (self.y1 != self.y2 and self.z1 != self.z2) else None
            self.right = torch.cat([torch.full((num**2, 1), self.x2), yz_y.reshape(-1, 1), yz_z.reshape(-1, 1)], dim=1).requires_grad_(True) if (self.y1 != self.y2 and self.z1 != self.z2) else None

        if (self.x1 != self.x2) and (self.y1 != self.y2) and (self.z1 != self.z2):
            faces = [self.bottom, self.top, self.front, self.back, self.left, self.right]
            self.plane = False
        else:
            faces = [self.bottom, self.front, self.left]
            self.plane = True
        return torch.cat([face for face in faces if face is not None], dim=0)


class sphere(geom3d):
    """
    sphere
    ======

    Provides
    1. A representation for 3D sphere geometry.
    2. Inherits from the `geom3d` abstract base class.

    How to use
    ----------
    Instantiate a sphere with the coordinates of the center and radius:

    >>> my_sphere = sphere(x, y, z, r)
    """
    def __init__(self, x, y, z, r):
        """
        Initializes a sphere object with given center coordinates and radius.

        Args:
        x (float): The x-coordinate of the center of the sphere.
        y (float): The y-coordinate of the center of the sphere.
        z (float): The z-coordinate of the center of the sphere.
        r (float): The radius of the sphere.
        """
        self.x = x
        self.y = y
        self.z = z
        self.r = r

    def generate_boundary_points(self, num=100):
        phi = torch.linspace(0, 2 * torch.pi, num)  # azimuthal angle
        theta = torch.linspace(0, torch.pi, num)  # polar angle

        phi, theta = torch.meshgrid(phi, theta, indexing='ij')
        phi = phi.reshape(-1,1)
        theta = theta.reshape(-1,1)

        # convert spherical coordinates to cartesian coordinates
        x = self.x + self.r * torch.sin(theta) * torch.cos(phi)
        y = self.y + self.r * torch.sin(theta) * torch.sin(phi)
        z = self.z + self.r * torch.cos(theta)

        return torch.cat([x.requires_grad_(True), y.requires_grad_(True), z.requires_grad_(True)], dim=1)

class disk(geom3d):
    def __init__(self, x, y, h, r, b=0, eps=2e-4):
        self.x = x
        self.y = y
        self.b = b
        self.h = h
        self.r = r
        self.eps = eps

    # the geom 'disk' only contains points on the side of the disk. The top and bottom faces are ignored.
    def generate_boundary_points(self, num=100):
        theta = torch.linspace(0, 2 * torch.pi, num)  # angle
        zz = torch.linspace(self.b + self.eps, self.h - self.eps, num)
        theta_, zz_ = torch.meshgrid(theta, zz, indexing='ij')
        theta_ = theta_.reshape(-1, 1)
        zz_ = zz_.reshape(-1, 1)
        # convert polar coordinates to cartesian coordinates
        x = self.x + self.r * torch.cos(theta_)
        y = self.y + self.r * torch.sin(theta_)
        z = zz_
        return torch.cat([x.requires_grad_(True), y.requires_grad_(True), z.requires_grad_(True)], dim=1)