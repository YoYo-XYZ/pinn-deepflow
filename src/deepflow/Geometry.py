import torch
from .Utility import *
from .Physics import PDE, NVS
import copy
from .PhysicsInformedAttach import PhysicsAttach

def string_to_grad(input:str):
    """""
    example use
    input: 'p'_'x'
    output: ['p','x']
    """
    character_list = input.split('_')
        
    return character_list

def rectangle(x_range:list , y_range:list):
    bound_list = [
        Bound(x_range, lambda x: y_range[0] * torch.ones_like(x), False, ref_axis='x'), # Bottom wall
        Bound(x_range, lambda x: y_range[1] * torch.ones_like(x), True, ref_axis='x'),  # Top wall
        Bound(y_range, lambda y: x_range[0] * torch.ones_like(y), False, ref_axis='y'), # Inlet
        Bound(y_range, lambda y: x_range[1] * torch.ones_like(y), True, ref_axis='y') # Outlet
    ]

    return Area(bound_list)

def circle(x, y, r):
    def func_up(X_tensor):
        return torch.sqrt(r**2 - (X_tensor-x)**2) + y
    def func_down(X_tensor):
        return -torch.sqrt(r**2 - (X_tensor-x)**2) + y
    
    def func_n_x_up(n):
        return x + r*torch.cos(n)
    def func_n_y_up(n):
        return y + r*torch.sin(n)
    def func_n_x_down(n):
        return x + r*torch.cos(n+torch.pi)
    def func_n_y_down(n):
        return y + r*torch.sin(n+torch.pi)
    
    bound_list = [
    Bound([x-r,x+r], func_up, True, ref_axis='x', func_n_x=func_n_x_up, func_n_y=func_n_y_up, range_n = [0,torch.pi]),
    Bound([x-r,x+r], func_down, False, ref_axis='x', func_n_x=func_n_x_down, func_n_y=func_n_y_down, range_n = [0,torch.pi])
    ]

    return Area(bound_list)

def line_horizontal(y, x_range):
    return Bound(x_range, lambda x: y * torch.ones_like(x), False, ref_axis='x')

def line_vertical(x, y_range):
    return Bound(y_range, lambda y: x * torch.ones_like(y), False, ref_axis='y')

class Bound(PhysicsAttach):
    def __init__(self, range_x, func_x, is_inside, ref_axis='x', func_n_x=None, func_n_y=None, range_n=None):
        super().__init__()
        self.range_x = range_x
        self.func_x = func_x
        self.is_inside = is_inside
        self.ref_axis = ref_axis
    
        self.func_n_x = func_n_x #for circle
        self.func_n_y = func_n_y
        self.range_n = range_n

        if ref_axis == 'x':
            range_true_x = range_x
            if self.func_n_x is not None:
                _, f_x = self.sampling_line(10000)
                f_x = f_x.detach().numpy()
            else:
                f_x = self.func_x(torch.linspace(self.range_x[0], self.range_x[1],10000)).detach().numpy()
            range_true_y = [float(min(f_x)),float(max(f_x))]
        elif ref_axis == 'y':
            range_true_y = range_x
            F_y = func_x(torch.linspace(range_x[0], range_x[1],1000)).detach().numpy()
            range_true_x = [float(min(F_y)), float(max(F_y))]

        self.length = range_true_x[1] - range_true_x[0]
        self.width = range_true_y[1] - range_true_y[0]

        self.x_center = range_true_x[0] + self.length / 2
        self.y_center = range_true_y[0] + self.width / 2

    def sampling_line(self, n_points, random=False):
        if self.func_n_x is None:
            if random:
                X = torch.empty(n_points).uniform_(self.range_x[0], self.range_x[1])
            else:
                X = torch.linspace(self.range_x[0], self.range_x[1], n_points)
            Y = self.func_x(X)
        else:
            if random:
                N = torch.empty(n_points).uniform_(self.range_n[0], self.range_n[1])
            else:
                N = torch.linspace(self.range_n[0], self.range_n[1], n_points)
            X = self.func_n_x(N)
            Y = self.func_n_y(N)

        if self.ref_axis == 'x':
            self.X = X
            self.Y = Y
            return X, Y
        else:
            self.X = Y
            self.Y = X
            return Y, X
    
    def mask_area(self, x, y):
        if self.ref_axis == 'y':
            x, y = y, x

        reject_mask_x = (self.range_x[0] < x) & (x < self.range_x[1])
        if self.is_inside:
            reject_mask_y = (y > self.func_x(x))
        else:
            reject_mask_y = (y < self.func_x(x))

        return reject_mask_x & reject_mask_y
class Area(PhysicsAttach):
    def __init__(self, bound_list: list[Bound], negative_bound_list:list[Bound] = None):
        super().__init__()
        self.bound_list = bound_list
        self.negative_bound_list = negative_bound_list

        range_x = []
        range_y = []
        for bound in bound_list:
            if bound.ref_axis == 'x':
                range_x += bound.range_x
                if bound.func_n_x is not None:
                    _, f_x = bound.sampling_line(10000)
                    f_x = f_x.detach().numpy()
                else:
                    f_x = bound.func_x(torch.linspace(bound.range_x[0], bound.range_x[1],10000)).detach().numpy()
                range_y += [float(min(f_x)),float(max(f_x))]
            elif bound.ref_axis == 'y':
                range_y += bound.range_x
                F_y = bound.func_x(torch.linspace(bound.range_x[0], bound.range_x[1],10000)).detach().numpy()
                range_x += [float(min(F_y)),float(max(F_y))]

        self.range_x = [min(range_x), max(range_x)]
        self.range_y = [min(range_y), max(range_y)]

        self.length = self.range_x[1] - self.range_x[0]
        self.width = self.range_y[1] - self.range_y[0]

        self.x_center = self.range_x[0] + self.length / 2
        self.y_center = self.range_y[0] + self.width / 2

    def sampling_area(self, n_points_square, random=False):
        if random:
            points = torch.empty(n_points_square, 2)
            points[:, 0].uniform_(self.range_x[0] + 1e-6, self.range_x[1] - 1e-6)  # x values
            points[:, 1].uniform_(self.range_y[0] + 1e-6, self.range_y[1] - 1e-6)  # y values
            X = points[:, 0]  # x-coordinates
            Y = points[:, 1]  # y-coordinates
        else:
            if isinstance(n_points_square, list):
                n_points_square_x = n_points_square[0]
                n_points_square_y = n_points_square[1]
            else:
                n_points_square_x = n_points_square_y = n_points_square
            X_range = torch.linspace(self.range_x[0]+1e-6, self.range_x[1]-1e-6, n_points_square_x)
            Y_range = torch.linspace(self.range_y[0]+1e-6, self.range_y[1]-1e-6, n_points_square_y)
            X, Y = torch.meshgrid(X_range, Y_range)
            X = X.reshape(-1)  # x-coordinates
            Y = Y.reshape(-1)  # y-coordinates

        reject_mask_list = []
        for bound in self.bound_list:
            reject_mask_list.append(bound.mask_area(X,Y))
        self.reject_mask = torch.stack(reject_mask_list, dim=0).any(dim=0)

        if self.negative_bound_list is not None:
            negative_reject_mask_list = []
            for bound in self.negative_bound_list:
                negative_reject_mask_list.append(bound.mask_area(X,Y))
            negative_reject_mask = torch.stack(negative_reject_mask_list, dim=0).all(dim=0)
            self.reject_mask = self.reject_mask | negative_reject_mask

        self.X, self.Y = X[~self.reject_mask], Y[~self.reject_mask]
        self.sampled_area = (self.X, self.Y)
        return self.X, self.Y

    def __sub__(self, other_area):
        bound_list = copy.deepcopy(other_area.bound_list)
        for bound in bound_list:
            bound.is_inside = not bound.is_inside
        return Area(self.bound_list, bound_list)

    def __add__(self, other_bound):
        X = torch.cat([self.sampled_area[0], other_bound.sampled_area[0]], dim=0)
        Y = torch.cat([self.sampled_area[1], other_bound.sampled_area[1]], dim=0)
        return X, Y