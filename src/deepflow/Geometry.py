import torch
from .Utility import *
import copy
from .PhysicsInformedAttach import PhysicsAttach

def circle(x, y, r):
    def func_up(X_tensor):
        return (r**2 - (X_tensor-x)**2)**0.5 + y
    def func_down(X_tensor):
        return -(r**2 - (X_tensor-x)**2)**0.5 + y
    def func_n_x_up(n):
        return x + r*torch.cos(n)
    def func_n_y_up(n):
        return y + r*torch.sin(n)
    def func_n_x_down(n):
        return x + r*torch.cos(n+torch.pi)
    def func_n_y_down(n):
        return y + r*torch.sin(n+torch.pi)
    
    bound_up = Bound([x - r, x + r], func_up, ref_axis='x')
    bound_up.define_func([0,torch.pi], func_n_x_up, func_n_y_up, ref_axis='t')

    bound_down = Bound([x - r, x + r], func_down, ref_axis='x')
    bound_down.define_func([0,torch.pi], func_n_x_down, func_n_y_down, ref_axis='t')

    return Area([bound_up, bound_down])

def rectangle(x_range:list , y_range:list):
    return polygon([x_range[0], y_range[0]], [x_range[1], y_range[0]], [x_range[1], y_range[1]], [x_range[0], y_range[1]])

def line_horizontal(y, range_x):
    return Bound(range_x, lambda x: y * torch.ones_like(x), ref_axis='x')

def line_vertical(x, range_y):
    bound = Bound(range_y, lambda y: x * torch.ones_like(y), ref_axis='y')
    bound.define_func([x- 1e-6, x +1e-6], lambda x_: 99999*(x_-x) + (range_y[0]+range_y[1])/2, ref_axis='x')
    return bound

def line(pos1:list, pos2:list):
    try:
        slope = (pos2[1] - pos1[1]) / (pos2[0] - pos1[0])
    except ZeroDivisionError:
        return line_vertical(pos1[0], sorted([pos2[1], pos1[1]]))
    return Bound(sorted([pos1[0], pos2[0]]), lambda x: slope * (x - pos1[0]) + pos1[1], ref_axis='x')

def polygon(*pos):
    bound_list = []
    for i, _ in enumerate(pos):
        bound_list.append(line(pos[i],pos[i-1]))
    return Area(bound_list)

class Bound(PhysicsAttach):
    dim = 2
    axes = list(range(dim))

    def define_func(self, range_, *func, ref_axis='y'):
        if ref_axis == 'x' :
            ax = 0
        elif ref_axis == 'y':
            ax = 1
        else:
            self.parameterized = True
            ax = 2
        self.funcs[ax] = list(func)
        self.ranges[ax] = sorted(range_)
        self._postprocess()

    def __init__(self, range_, *func , ref_axis = 'x'):
        super().__init__()
        self.ranges = {}
        self.funcs = {}
        self.coords = {}
        self.parameterized = False
        if ref_axis == 'x' :
            self.ax = 0
        elif ref_axis == 'y':
            self.ax = 1
        else:
            self.ax = 2

        self.axes_sec = list(self.axes)
        if self.ax in self.axes_sec:
            self.axes_sec.remove(self.ax)

        self.ranges[self.ax] = sorted(range_)
        self.funcs[self.ax] = list(func) #func is list
        self.reject_above = True
        self._postprocess()

    def _postprocess(self):
        self.sampling_line(10000, random=False)
        self.lengths = {}
        self.centers = {}
        for ax in self.axes_sec:
            self.ranges[ax] = [self.coords[ax].min().item(), self.coords[ax].max().item()]
            # secondary varibales
        for ax in self.axes:
            self.lengths[ax] = self.ranges[ax][1] - self.ranges[ax][0]
            self.centers[ax] = self.ranges[ax][0] + self.lengths[ax]/2

    def sampling_line(self, n_points, random=False):
        ax = 2 if self.parameterized else self.ax
        self.coords = {}
        if random:
            self.coords[ax] = torch.empty(n_points).uniform_(self.ranges[ax][0], self.ranges[ax][1])
        else:
            self.coords[ax] = torch.linspace(self.ranges[ax][0], self.ranges[ax][1], n_points)
        if self.parameterized:
            self.coords[0] = self.funcs[2][0](self.coords[ax])
            self.coords[1] = self.funcs[2][1](self.coords[ax])
        else:
            for i, axis in enumerate(self.axes_sec):
                self.coords[axis] = self.funcs[ax][i](self.coords[ax])
        return self.coords[0], self.coords[1]

    def mask_area(self, *x: torch.Tensor):
        reject_masks = []
        self.ranges[self.ax] = self.ranges[self.ax]
        reject_masks.append((x[self.ax] > self.ranges[self.ax][0]) & (x[self.ax] < self.ranges[self.ax][1]))
        for sec_ax in self.axes_sec:
            if self.reject_above:
                reject_masks.append((x[sec_ax] >= self.funcs[self.ax][0](x[self.ax])))
            else:
                reject_masks.append((x[sec_ax] <= self.funcs[self.ax][0](x[self.ax])))
        return reject_masks[0] & reject_masks[1]

    def __add__(self, other_bound):
        bound_list = [self, other_bound]
        return Area(bound_list)
    
    def __str__(self):
        return f'axis: {self.ax} reject above: {self.reject_above}, ranges: {self.ranges}, centers: {self.centers}, lengths: {self.lengths}'
    
    def show(self):
        import matplotlib.pyplot as plt
        X, Y = self.sampling_line(1000)
        plt.plot(X.numpy(), Y.numpy())
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

class Area(PhysicsAttach):
    dim = 2
    axes = list(range(dim))
    def __init__(self, bound_list: list[Bound], bounds_negative:list[Bound] = None):
        super().__init__()
        self.ranges = {}
        self.bound_list = bound_list
        self.negative_bound_list = bounds_negative
        
        for ax in self.axes:
            range_list = []
            for bound in bound_list:
                range_list += bound.ranges[ax]
            self.ranges[ax] = (min(range_list), max(range_list))
        self._postprocess()
        self.checkbound()

    def _postprocess(self):
        self.lengths = {}
        self.centers = {}
        for ax in self.axes:
            self.lengths[ax] = self.ranges[ax][1] - self.ranges[ax][0]
            self.centers[ax] = self.ranges[ax][0] + self.lengths[ax]/2

    def checkbound(self):
        bound_list = self.bound_list
        def is_inrange(x, range_x):
            return range_x[0] < x < range_x[1]
        #brute force checking
        ax = 0
        for i, bound in enumerate(bound_list):
            if bound.ax == ax:
                x = bound.centers[ax]
                # assign reject_above based on sorted y-values at x_center
                y_dict = {}
                for j, bound_opponent in enumerate(bound_list):
                    # create sorted y_value dict
                    if is_inrange(x, bound_opponent.ranges[ax]):
                        y_dict[j] = bound_opponent.funcs[ax][0](x)
                    sorted_index =  [index for (index, y) in sorted(y_dict.items(), key=lambda item: item[1])]
                    # assign reject_above based on sorted y-values
                    for jj, index in enumerate(sorted_index):
                        if jj % 2 == 0:
                            bound_list[index].reject_above = False
                        else:
                            bound_list[index].reject_above = True
            else:
                x = bound.centers[ax]+ 1e-4
                # assign reject_above based on sorted y-values at x_center
                y_dict = {}
                for j, bound_opponent in enumerate(bound_list):
                    # create sorted y_value dict
                    if is_inrange(x, bound_opponent.ranges[ax]):
                        y_dict[j] = bound_opponent.funcs[ax][0](x)
                    sorted_index =  [index for (index, y) in sorted(y_dict.items(), key=lambda item: item[1])]
                    # assign reject_above based on sorted y-values
                    if sorted_index:
                        for jj, index in enumerate(sorted_index):
                            if round(y_dict[index],2) == round(bound.ranges[1][0],2):
                                bound.reject_above = False if jj % 2 == 0 else True
                            elif round(y_dict[index],2) == round(bound.ranges[1][1],2):
                                bound.reject_above = True if jj % 2 == 0 else False
                    else:
                        bound.reject_above = True

    def sampling_area(self, n_points_square, random=False):
        if random:
            points = torch.empty(n_points_square, 2)
            points[:, 0].uniform_(self.ranges[0][0], self.ranges[0][1])  # x values
            points[:, 1].uniform_(self.ranges[1][0], self.ranges[1][1])  # y values
            X = points[:, 0]  # x-coordinates
            Y = points[:, 1]  # y-coordinates
        else:
            if isinstance(n_points_square, list):
                n_points_square_x = n_points_square[0]
                n_points_square_y = n_points_square[1]
            else:
                n_points_square_x = n_points_square_y = n_points_square
            X_range = torch.linspace(self.ranges[0][0], self.ranges[0][1], n_points_square_x)
            X_range[0] += 1e-6
            X_range[-1] -= 1e-6
            Y_range = torch.linspace(self.ranges[1][0], self.ranges[1][1], n_points_square_y)
            Y_range[0] += 1e-6
            Y_range[-1] -= 1e-6
            X, Y = torch.meshgrid(X_range, Y_range, indexing='ij')
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

    def sampling_lines(self, *n_points_per_line, random=False):
        output_x = []
        output_y = []
        if len(n_points_per_line) == 1:
            n_points_per_line = [n_points_per_line[0]] * len(self.bound_list)
        for i, bound in enumerate(self.bound_list):
            X,Y = bound.sampling_line(n_points_per_line[i], random=random)
            output_x.append(X)
            output_y.append(Y)
        output_x = [torch.stack(output_x, dim=0)]
        output_y = [torch.stack(output_y, dim=0)]
        return output_x, output_y

    def __sub__(self, other_area):
        bound_list = copy.deepcopy(other_area.bound_list)
        for bound in bound_list:
            bound.reject_above = not bound.reject_above
        return Area(self.bound_list, bound_list)

    def __add__(self, other_bound):
        bound_list_new = self.bound_list.copy().append(other_bound)
        return Area(bound_list_new)
    
    def __iter__(self):
        return iter(self.bound_list)
    
    def __str__(self):
        string = ''
        for i, bound in enumerate(self.bound_list):
            string += f"{i} {bound}\n"
        string += f"ranges: {self.ranges}"
        return string
    
    def show(self, show_index=False):
        import matplotlib.pyplot as plt
        for bound in self.bound_list:
            X, Y = bound.sampling_line(1000)
            plt.plot(X.numpy(), Y.numpy())
            if show_index:
                plt.text(bound.centers[0], bound.centers[1], f'{self.bound_list.index(bound)}')
        for bound in self.negative_bound_list or []:
            X, Y = bound.sampling_line(1000)
            plt.plot(X.numpy(), Y.numpy(), color='red')
        X, Y = self.sampling_area([100,100], random=False)
        plt.scatter(X.numpy(), Y.numpy(), s=0.05, color='green')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        return X, Y
    
def shape(bound_list: list[Bound]):
    return Area(bound_list)

def curve(range_, *func , ref_axis = 'x'):
    return Bound(range_, *func , ref_axis = ref_axis)