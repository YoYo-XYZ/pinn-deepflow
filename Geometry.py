import copy
from typing import List, Tuple, Callable, Optional, Union, Dict

import torch
import numpy as np # Often useful to have explicit numpy when dealing with plotting
from .Utility import *  # Ideally, import specific functions: from .Utility import func1, func2
from .PhysicsInformedAttach import PhysicsAttach

# Constants for numerical stability
EPS = 1e-6
LARGE_SLOPE = 1e5

class Bound(PhysicsAttach):
    """
    Represents a 1D boundary in a 2D space (e.g., a line segment or curve).
    
    Attributes:
        ranges (Dict): Dictionary storing min/max values for axes.
        funcs (Dict): Dictionary storing parameterization functions.
    """
    dim = 2
    axes = list(range(dim))

    def __init__(self, range_val: List[float], *func: Callable, ref_axis: str = 'x'):
        """
        Initialize a Boundary.

        Args:
            range_val: The [min, max] range along the reference axis.
            *func: Functions defining the geometry.
            ref_axis: The axis on which the range is defined ('x', 'y', or 't' for parametric).
        """
        super().__init__()
        self.ranges: Dict[int, List[float]] = {}
        self.funcs: Dict[int, List[Callable]] = {}
        self.coords: Dict[int, torch.Tensor] = {}
        self.parameterized = False
        
        # Determine axis index
        if ref_axis == 'x':
            self.ax = 0
        elif ref_axis == 'y':
            self.ax = 1
        else:
            self.ax = 2  # Parametric 't'

        self.axes_sec = list(self.axes)
        if self.ax in self.axes_sec:
            self.axes_sec.remove(self.ax)

        self.ranges[self.ax] = sorted(range_val)
        self.funcs[self.ax] = list(func)
        self.reject_above = True
        self._postprocess()

    def define_func(self, range_val: List[float], *func: Callable, ref_axis: str = 'y'):
        """Defines secondary functions or parameterization for the boundary."""
        if ref_axis == 'x':
            ax = 0
        elif ref_axis == 'y':
            ax = 1
        else:
            self.parameterized = True
            ax = 2
            
        self.funcs[ax] = list(func)
        self.ranges[ax] = sorted(range_val)
        self._postprocess()

    def _postprocess(self):
        """Calculates lengths and centers after definition."""
        # Preliminary sampling to determine bounds of dependent axes
        self.sampling_line(10000, random=False)
        self.lengths = {}
        self.centers = {}
        
        for ax in self.axes_sec:
            self.ranges[ax] = [self.coords[ax].min().item(), self.coords[ax].max().item()]
            
        for ax in self.axes:
            self.lengths[ax] = self.ranges[ax][1] - self.ranges[ax][0]
            self.centers[ax] = self.ranges[ax][0] + self.lengths[ax] / 2

    def sampling_line(self, n_points: int, random: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates points along the boundary.

        Args:
            n_points: Number of points to generate.
            random: If True, samples uniformly; otherwise, uses linspace.
        """
        ax = 2 if self.parameterized else self.ax
        self.coords = {}
        
        if random:
            self.coords[ax] = torch.empty(n_points).uniform_(self.ranges[ax][0], self.ranges[ax][1])
        else:
            self.coords[ax] = torch.linspace(self.ranges[ax][0], self.ranges[ax][1], n_points)

        if self.parameterized:
            # Assuming funcs[2] contains [func_x(t), func_y(t)]
            self.coords[0] = self.funcs[2][0](self.coords[ax])
            self.coords[1] = self.funcs[2][1](self.coords[ax])
        else:
            for i, axis in enumerate(self.axes_sec):
                self.coords[axis] = self.funcs[ax][i](self.coords[ax])
                
        self.X, self.Y = self.coords[0], self.coords[1]
        return self.X, self.Y

    def mask_area(self, *x: torch.Tensor) -> torch.Tensor:
        """Creates a boolean mask for points relative to this boundary."""
        reject_masks = []
        
        # Check if within the reference axis range
        in_range_mask = (x[self.ax] > self.ranges[self.ax][0]) & (x[self.ax] < self.ranges[self.ax][1])
        reject_masks.append(in_range_mask)
        
        for sec_ax in self.axes_sec:
            boundary_val = self.funcs[self.ax][0](x[self.ax])
            if self.reject_above:
                reject_masks.append(x[sec_ax] >= boundary_val)
            else:
                reject_masks.append(x[sec_ax] <= boundary_val)
                
        return reject_masks[0] & reject_masks[1]

    def __add__(self, other_bound: 'Bound') -> 'Area':
        return Area([self, other_bound])
    
    def __str__(self):
        return (f'Bound(axis={self.ax}, reject_above={self.reject_above}, '
                f'ranges={self.ranges})')
    
    def show(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is required to use show()")
            return

        X, Y = self.sampling_line(1000)
        plt.plot(X.numpy(), Y.numpy())
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


class Area(PhysicsAttach):
    """
    Represents a 2D area defined by a collection of Boundaries.
    """
    dim = 2
    axes = list(range(dim))

    def __init__(self, bound_list: List[Bound], bounds_negative: Optional[List[Bound]] = None):
        super().__init__()
        self.ranges = {}
        self.bound_list = bound_list
        self.negative_bound_list = bounds_negative
        
        # Calculate bounding box for the entire area
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
            self.centers[ax] = self.ranges[ax][0] + self.lengths[ax] / 2

    def checkbound(self):
        """
        Automatically determines the 'direction' (reject_above) of boundaries
        using a ray-casting approach from the center.
        """
        def is_inrange(val, rng):
            return rng[0] < val < rng[1]

        # Brute force checking orientation
        for bound in self.bound_list:
            ax = bound.ax
            
            if ax in [0, 1]: # Standard x or y axis definition
                x = bound.centers[ax]
            else: # Parameterized
                x = bound.centers[0] + 1e-4

            # Collect intersection points with other bounds at this coordinate
            y_dict = {}
            for j, bound_opponent in enumerate(self.bound_list):
                # Ensure we handle the correct axis lookup for the opponent
                opp_ax = bound_opponent.ax if bound_opponent.ax < 2 else 0
                
                if is_inrange(x, bound_opponent.ranges[opp_ax]):
                     # Note: This assumes simple functions. Complex parameterizations may fail here.
                    try:
                        val = bound_opponent.funcs[opp_ax][0](torch.tensor(x)).item()
                        y_dict[j] = val
                    except:
                        continue

            sorted_items = sorted(y_dict.items(), key=lambda item: item[1])
            sorted_indices = [idx for idx, _ in sorted_items]

            # Assign rejection logic based on even/odd rule (standard for polygons)
            for jj, index in enumerate(sorted_indices):
                # Logic: If I am the boundary at this location...
                if self.bound_list[index] == bound:
                    # Using isclose for float comparison instead of round()
                    is_even = (jj % 2 == 0)
                    bound.reject_above = not is_even

    def sampling_area(self, n_points_square: Union[int, List[int]], random: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples points within the area.
        """
        if random:
            # Rejection sampling initialization (oversample slightly to ensure density)
            n_pts = n_points_square if isinstance(n_points_square, int) else n_points_square[0] * n_points_square[1]
            points = torch.empty(n_pts, 2)
            points[:, 0].uniform_(self.ranges[0][0], self.ranges[0][1])
            points[:, 1].uniform_(self.ranges[1][0], self.ranges[1][1])
            X, Y = points[:, 0], points[:, 1]
        else:
            if isinstance(n_points_square, list):
                nx, ny = n_points_square[0], n_points_square[1]
            else:
                nx = ny = n_points_square
                
            X_range = torch.linspace(self.ranges[0][0], self.ranges[0][1], nx)
            Y_range = torch.linspace(self.ranges[1][0], self.ranges[1][1], ny)
            
            # Padding to avoid hitting exact boundaries
            X_range[0] += EPS
            X_range[-1] -= EPS
            Y_range[0] += EPS
            Y_range[-1] -= EPS
            
            X, Y = torch.meshgrid(X_range, Y_range, indexing='ij')
            X = X.reshape(-1)
            Y = Y.reshape(-1)

        # Apply Masks
        reject_mask_list = [bound.mask_area(X, Y) for bound in self.bound_list]
        self.reject_mask = torch.stack(reject_mask_list, dim=0).any(dim=0)

        if self.negative_bound_list is not None:
            neg_masks = [b.mask_area(X, Y) for b in self.negative_bound_list]
            negative_reject_mask = torch.stack(neg_masks, dim=0).all(dim=0)
            self.reject_mask = self.reject_mask | negative_reject_mask
            
        self.X, self.Y = X[~self.reject_mask], Y[~self.reject_mask]
        self.sampled_area = (self.X, self.Y)
        return self.X, self.Y

    def sampling_lines(self, *n_points_per_line, random: bool = False):
        output_x = []
        output_y = []
        
        pts_list = list(n_points_per_line)
        if len(pts_list) == 1:
            pts_list = [pts_list[0]] * len(self.bound_list)
            
        for i, bound in enumerate(self.bound_list):
            num_pts = pts_list[i] if i < len(pts_list) else pts_list[0]
            X, Y = bound.sampling_line(num_pts, random=random)
            output_x.append(X)
            output_y.append(Y)
            
        return [torch.stack(output_x, dim=0)], [torch.stack(output_y, dim=0)]

    def __sub__(self, other_area: 'Area') -> 'Area':
        """Boolean subtraction of geometry."""
        bound_list = copy.deepcopy(other_area.bound_list)
        for bound in bound_list:
            bound.reject_above = not bound.reject_above
        return Area(self.bound_list, bound_list)

    def __add__(self, other_bound: Union['Bound', 'Area']) -> 'Area':
        """Combine areas."""
        if isinstance(other_bound, Area):
            # Logic to merge two areas would go here
            pass
        new_list = self.bound_list.copy()
        new_list.append(other_bound)
        return Area(new_list)
    
    def __iter__(self):
        return iter(self.bound_list)
    
    def __str__(self):
        s = "Area defined by bounds:\n"
        for i, bound in enumerate(self.bound_list):
            s += f"  {i}: {bound}\n"
        s += f"  Global Range: {self.ranges}"
        return s
    
    def show(self, show_index: bool = False):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        for bound in self.bound_list:
            X, Y = bound.sampling_line(1000)
            plt.plot(X.numpy(), Y.numpy(), color='blue')
            if show_index:
                plt.text(bound.centers[0], bound.centers[1], f'{self.bound_list.index(bound)}')
                
        for bound in self.negative_bound_list or []:
            X, Y = bound.sampling_line(1000)
            plt.plot(X.numpy(), Y.numpy(), color='red', linestyle='--')
            
        # Sample interior to verify logic
        X, Y = self.sampling_area([100, 100], random=False)
        plt.scatter(X.numpy(), Y.numpy(), s=0.05, color='green', alpha=0.5)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

# --- Factory Functions ---

def circle(x: float, y: float, r: float) -> Area:
    """Creates a circular Area."""
    def func_up(X_tensor):
        return (r**2 - (X_tensor - x)**2)**0.5 + y
    def func_down(X_tensor):
        return -(r**2 - (X_tensor - x)**2)**0.5 + y
    
    # Parametric definitions for plotting/sampling
    def func_n_x_up(n): return x + r * torch.cos(n)
    def func_n_y_up(n): return y + r * torch.sin(n)
    def func_n_x_down(n): return x + r * torch.cos(n + torch.pi)
    def func_n_y_down(n): return y + r * torch.sin(n + torch.pi)
    
    bound_up = Bound([x - r, x + r], func_up, ref_axis='x')
    bound_up.define_func([0, torch.pi], func_n_x_up, func_n_y_up, ref_axis='t')

    bound_down = Bound([x - r, x + r], func_down, ref_axis='x')
    bound_down.define_func([0, torch.pi], func_n_x_down, func_n_y_down, ref_axis='t')

    return Area([bound_up, bound_down])

def rectangle(x_range: List[float], y_range: List[float]) -> Area:
    """Creates a rectangular Area."""
    p1 = [x_range[0], y_range[0]]
    p2 = [x_range[1], y_range[0]]
    p3 = [x_range[1], y_range[1]]
    p4 = [x_range[0], y_range[1]]
    return polygon(p1, p2, p3, p4)

def line_horizontal(y: float, range_x: List[float]) -> Bound:
    return Bound(range_x, lambda x: y * torch.ones_like(x), ref_axis='x')

def line_vertical(x: float, range_y: List[float]) -> Bound:
    bound = Bound(range_y, lambda y: x * torch.ones_like(y), ref_axis='y')
    # Using a large slope to approximate verticality for x-based lookups
    bound.define_func([x - EPS, x + EPS], 
                      lambda x_: LARGE_SLOPE * (x_ - x) + (range_y[0] + range_y[1]) / 2, 
                      ref_axis='x')
    return bound

def line(pos1: List[float], pos2: List[float]) -> Bound:
    """Creates a boundary line between two points."""
    x1, y1 = pos1
    x2, y2 = pos2
    
    if abs(x2 - x1) < EPS:
        return line_vertical(x1, sorted([y1, y2]))
        
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Note: capturing slope/intercept in lambda default args to avoid late binding issues
    return Bound(sorted([x1, x2]), 
                 lambda x, m=slope, c=intercept: m * x + c, 
                 ref_axis='x')

def polygon(*pos: List[float]) -> Area:
    """
    Creates a polygon from a sequence of vertex coordinates.
    Vertices should be ordered (clockwise or counter-clockwise).
    """
    bound_list = []
    for i in range(len(pos)):
        # Connect current point to the previous point
        bound_list.append(line(pos[i], pos[i-1]))
    return Area(bound_list)

def curve(range_val: List[float], *func, ref_axis='x') -> Bound:
    return Bound(range_val, *func, ref_axis=ref_axis)