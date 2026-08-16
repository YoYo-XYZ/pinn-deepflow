"""Two-dimensional geometry primitives and sampling helpers."""

from typing import List, Tuple, Callable, Optional, Union, Dict, Any

import torch

from .utility import *
from .utility import _next_seed, get_dtype
from .physicsinformed import PhysicsAttach

# Constants for numerical stability
EPS = 1e-6
LARGE_SLOPE = 1e5

def custom_data(data_dict: Dict[str, Any]) -> 'CustomData':
    """Create a geometry backed by user-provided coordinate data.

    Args:
        data_dict: Mapping containing at least ``x`` and ``y`` coordinate
            tensors. Additional entries are retained as user data.

    Returns:
        A ``CustomData`` instance representing the supplied points.
    """
    return CustomData(data_dict)
class CustomData(PhysicsAttach):
    """Attach PINN physics and evaluation to a fixed point cloud.

    Args:
        data_dict: Mapping containing ``x`` and ``y`` coordinate tensors.

    Notes:
        Custom data is not resampled by area or line samplers. Its coordinates
        are processed as supplied.
    """
    dim = 2
    axes = list(range(dim))
    
    def __init__(self, data_dict: Dict[str, Any]):
        """Initialize a custom-data geometry from coordinate arrays."""
        super().__init__()
        self.ranges: Dict[int, List[float]] = {}
        self.axes_sec = list(self.axes)
        self.X = data_dict.get('x')
        self.Y = data_dict.get('y')
        
        self.data_dict = data_dict
        self.coords = {i: data_dict.get(ax) for i, ax in enumerate(['x', 'y'])}
        self._postprocess()

    def _postprocess(self):
        """Calculates lengths and centers after definition."""
        # Preliminary sampling to determine bounds of dependent axes
        self.lengths = {}
        self.centers = {}
        
        for ax in self.axes_sec:
            self.ranges[ax] = [self.coords[ax].min().item(), self.coords[ax].max().item()]
            
        for ax in self.axes:
            self.lengths[ax] = self.ranges[ax][1] - self.ranges[ax][0]
            self.centers[ax] = self.ranges[ax][0] + self.lengths[ax] / 2

class Bound(PhysicsAttach):
    """Represent a one-dimensional boundary in a two-dimensional space.

    A boundary is described by a reference-axis interval and one or more
    functions for the dependent coordinates. Use ``line``, ``curve``, or
    ``point`` for common constructions.

    Args:
        range_val: Two-element interval on the reference axis.
        *func: Functions mapping the reference coordinate to dependent
            coordinates, or two functions mapping a parameter to ``x`` and
            ``y`` when ``ref_axis="t"``.
        ref_axis: Reference axis: ``"x"``, ``"y"``, or ``"t"``.

    Attributes:
        ranges: Minimum and maximum values for each coordinate axis.
        funcs: Parameterization functions keyed by reference axis.
    """
    dim = 2
    axes = list(range(dim))

    def __init__(self, range_val: List[float], *func: Callable, ref_axis: str = 'x'):
        """Initialize a boundary parameterization."""
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
        """Define or replace a dependent-coordinate parameterization.

        Args:
            range_val: Interval for the selected reference axis.
            *func: Functions defining dependent coordinates or a parametric
                ``x(t), y(t)`` curve.
            ref_axis: Reference axis: ``"x"``, ``"y"``, or ``"t"``.

        Returns:
            None. The boundary bounds and sampled preview are updated in place.
        """
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
        self.sampling_line(10000, scheme='uniform')
        self.lengths = {}
        self.centers = {}
        
        for ax in self.axes_sec:
            self.ranges[ax] = [self.coords[ax].min().item(), self.coords[ax].max().item()]
            
        for ax in self.axes:
            self.lengths[ax] = self.ranges[ax][1] - self.ranges[ax][0]
            self.centers[ax] = self.ranges[ax][0] + self.lengths[ax] / 2

    def sampling_line(self, n_points: int, scheme = 'uniform') -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points along the boundary.

        Args:
            n_points: Number of points to generate.
            scheme: Sampling scheme: ``"uniform"``, ``"random"``, or
                ``"lhs"`` for Latin Hypercube Sampling.

        Returns:
            Tuple of one-dimensional ``x`` and ``y`` tensors.
        """
        ax = 2 if self.parameterized else self.ax
        self.coords = {}
        
        if scheme == 'random':
            gen_seed = _next_seed()
            if gen_seed is not None:
                gen = torch.Generator().manual_seed(gen_seed)
                self.coords[ax] = torch.empty(n_points, dtype=get_dtype()).uniform_(self.ranges[ax][0], self.ranges[ax][1], generator=gen)
            else:
                self.coords[ax] = torch.empty(n_points, dtype=get_dtype()).uniform_(self.ranges[ax][0], self.ranges[ax][1])
        elif scheme == 'lhs':
            self.coords[ax] = latin_hypercube_sampling(n_points, 1, [self.ranges[ax][0]], [self.ranges[ax][1]]).squeeze(-1)
        elif scheme == 'uniform':
            self.coords[ax] = torch.linspace(self.ranges[ax][0], self.ranges[ax][1], n_points, dtype=get_dtype())

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
        """Return the mask of points rejected by this boundary.

        Args:
            *x: Coordinate tensors in ``x``/``y`` axis order.

        Returns:
            Boolean tensor with the same shape as the coordinate tensors.
        """
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
        return (f'Bound(axis={self.ax}, reject_above={self.reject_above}, ranges={self.ranges}), centers: {self.centers}, lengths: {self.lengths}')
    
    def show(self):
        """Plot a sampled representation of the boundary."""
        import matplotlib.pyplot as plt

        X, Y = self.sampling_line(1000)
        plt.plot(X.numpy(), Y.numpy())
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

class Area(PhysicsAttach):
    """Represent a two-dimensional area bounded by ``Bound`` objects.

    Args:
        bound_list: Boundary objects describing the exterior of the area.
        bounds_negative: Optional boundaries describing holes to exclude.
        contains_fn: Optional custom point-containment function.
        ranges: Optional explicit ``{axis: (lower, upper)}`` bounding ranges.

    Notes:
        Areas support subtraction and union with ``-``, ``|``, and ``+``.
        ``sampling_area`` filters candidate points through ``contains``.
    """
    dim = 2
    axes = list(range(dim))

    def __init__(
        self,
        bound_list: List[Bound],
        bounds_negative: Optional[List[Bound]] = None,
        *,
        contains_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        ranges: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        """Initialize an area from its boundary and containment definition."""
        super().__init__()
        self.bound_list = bound_list
        self.negative_bound_list = bounds_negative
        self._contains_fn = contains_fn
        
        if ranges is None:
            self.ranges = {}
            for ax in self.axes:
                range_list = []
                for bound in bound_list:
                    range_list += bound.ranges[ax]
                self.ranges[ax] = (min(range_list), max(range_list))
        else:
            self.ranges = ranges
            
        self._postprocess()
        if self._contains_fn is None:
            self.checkbound()

    def _postprocess(self):
        self.lengths = {}
        self.centers = {}
        for ax in self.axes:
            self.lengths[ax] = self.ranges[ax][1] - self.ranges[ax][0]
            self.centers[ax] = self.ranges[ax][0] + self.lengths[ax] / 2

    def checkbound(self):
        """
        Determine each boundary's interior-facing direction.

        The direction is inferred using a ray-casting check from the area
        center and stored on each boundary as ``reject_above``.

        Returns:
            None. Boundary objects are updated in place.
        """
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

    def contains(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask selecting points inside or on the area.

        Args:
            x: Floating-point tensor of x coordinates.
            y: Floating-point tensor of y coordinates with the same shape,
                device, and dtype as ``x``.

        Returns:
            Boolean tensor indicating whether each point belongs to the area.

        Raises:
            TypeError: If coordinates are not compatible floating-point
                tensors or a custom containment function returns an invalid
                value.
            ValueError: If coordinate shapes, devices, or dtypes differ.
        """
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("x and y must be torch tensors")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError("x and y must have the same device and dtype")
        if not x.is_floating_point():
            raise TypeError("x and y must be floating-point tensors")

        if self._contains_fn is not None:
            mask = self._contains_fn(x, y)
        else:
            reject_masks = [bound.mask_area(x, y) for bound in self.bound_list]
            mask = ~torch.stack(reject_masks, dim=0).any(dim=0)

            if self.negative_bound_list is not None:
                neg_masks = [bound.mask_area(x, y) for bound in self.negative_bound_list]
                mask &= ~torch.stack(neg_masks, dim=0).all(dim=0)

        if not isinstance(mask, torch.Tensor):
            raise TypeError("contains function must return a torch tensor")
        if mask.shape != x.shape or mask.dtype != torch.bool:
            raise TypeError(
                "contains function must return a boolean tensor with the same shape as x and y"
            )
        return mask

    def sampling_area(self, n_points_square: Union[int, List[int]], scheme = 'uniform') -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample and retain points inside the area.

        Args:
            n_points_square: For uniform sampling, an integer gives the
                number of points per axis; for random and LHS sampling, it
                gives the number of candidate points. ``[nx, ny]`` supplies
                separate grid resolutions and generates ``nx * ny``
                candidates.
            scheme: Sampling scheme: ``"uniform"``, ``"random"``, or
                ``"lhs"``.

        Returns:
            Tuple of sampled ``x`` and ``y`` tensors. Candidate points outside
            the area are removed.
        """
        # Normalize n_points_square to nx, ny, and n_total before branching
        if isinstance(n_points_square, (list, tuple)):
            nx, ny = n_points_square[0], n_points_square[1]
            n_total = nx * ny
        else:
            nx = ny = n_points_square
            n_total = n_points_square

        if scheme == 'random':
            gen_seed = _next_seed()
            if gen_seed is not None:
                gen = torch.Generator().manual_seed(gen_seed)
                points = torch.empty(n_total, 2, dtype=get_dtype())
                points[:, 0].uniform_(self.ranges[0][0], self.ranges[0][1], generator=gen)
                points[:, 1].uniform_(self.ranges[1][0], self.ranges[1][1], generator=gen)
            else:
                points = torch.empty(n_total, 2, dtype=get_dtype())
                points[:, 0].uniform_(self.ranges[0][0], self.ranges[0][1])
                points[:, 1].uniform_(self.ranges[1][0], self.ranges[1][1])
            X, Y = points[:, 0], points[:, 1]
        elif scheme == 'lhs':
            samples = latin_hypercube_sampling(n_total, 2, [self.ranges[0][0], self.ranges[1][0]], [self.ranges[0][1], self.ranges[1][1]]).squeeze(-1)
            X, Y = samples[:, 0], samples[:, 1]
        elif scheme == 'uniform':
            X_range = torch.linspace(self.ranges[0][0], self.ranges[0][1], nx, dtype=get_dtype())
            Y_range = torch.linspace(self.ranges[1][0], self.ranges[1][1], ny, dtype=get_dtype())
            
            # Padding to avoid hitting exact boundaries
            X_range[0] += EPS
            X_range[-1] -= EPS
            Y_range[0] += EPS
            Y_range[-1] -= EPS
            
            X, Y = torch.meshgrid(X_range, Y_range, indexing='ij')
            X = X.reshape(-1)
            Y = Y.reshape(-1)

        self.reject_mask = ~self.contains(X, Y)
            
        self.X, self.Y = X[~self.reject_mask], Y[~self.reject_mask]
        self.sampled_area = (self.X, self.Y)
        return self.X, self.Y

    def sampling_lines(self, *n_points_per_line, scheme = 'random') -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample every boundary in the area.

        Args:
            *n_points_per_line: One resolution applied to every boundary. If
                one value per boundary is supplied, all values must match so
                the result can be returned as dense stacked tensors.
            scheme: Sampling scheme passed to ``Bound.sampling_line``.

        Returns:
            Stacked ``x`` and ``y`` tensors containing the boundary samples.

        Raises:
            ValueError: If no resolution is supplied, the number of
                resolutions does not match the number of boundaries, or
                resolutions differ between boundaries.
        """
        if not self.bound_list:
            raise ValueError("Area has no boundaries to sample.")

        output_x = []
        output_y = []
        
        pts_list = list(n_points_per_line)
        if not pts_list:
            raise ValueError("At least one boundary sampling resolution is required.")
        if len(pts_list) == 1:
            pts_list = [pts_list[0]] * len(self.bound_list)
        elif len(pts_list) != len(self.bound_list):
            raise ValueError(
                "Provide one resolution for all boundaries or exactly one "
                "resolution per boundary."
            )
        if len(set(pts_list)) != 1:
            raise ValueError(
                "All boundary sampling resolutions must match when returning "
                "stacked tensors."
            )
            
        for i, bound in enumerate(self.bound_list):
            num_pts = pts_list[i]
            X, Y = bound.sampling_line(num_pts, scheme=scheme)
            output_x.append(X)
            output_y.append(Y)
            
        return torch.stack(output_x, dim=0), torch.stack(output_y, dim=0)

    def __sub__(self, other_area: 'Area') -> 'Area':
        """Boolean subtraction of geometry."""
        if not isinstance(other_area, Area):
            return NotImplemented

        negative_bound_list = list(self.negative_bound_list or [])
        negative_bound_list.extend(other_area.bound_list)

        return Area(
            self.bound_list.copy(),
            negative_bound_list,
            contains_fn=lambda x, y: self.contains(x, y) & ~other_area.contains(x, y),
            ranges=self.ranges.copy(),
        )

    def __or__(self, other: Union['Area', 'Bound']) -> 'Area':
        """Union areas, or attach a boundary without changing membership."""
        if isinstance(other, Bound):
            return self + other
        if not isinstance(other, Area):
            return NotImplemented

        ranges = {
            ax: (
                min(self.ranges[ax][0], other.ranges[ax][0]),
                max(self.ranges[ax][1], other.ranges[ax][1]),
            )
            for ax in self.axes
        }
        negative_bound_list = [
            *(self.negative_bound_list or []),
            *(other.negative_bound_list or []),
        ]

        return Area(
            [*self.bound_list, *other.bound_list],
            negative_bound_list or None,
            contains_fn=lambda x, y: self.contains(x, y) | other.contains(x, y),
            ranges=ranges,
        )

    def __add__(self, other: Union['Area', 'Bound']) -> 'Area':
        """Union areas, or retain the legacy boundary-addition behavior."""
        if isinstance(other, Area):
            return self | other
        if isinstance(other, Bound):
            new_bound_list = self.bound_list.copy()
            new_bound_list.append(other)
            return Area(
                new_bound_list,
                list(self.negative_bound_list or []) or None,
                contains_fn=self.contains,
                ranges=self.ranges.copy(),
            )
        return NotImplemented
    
    def __iter__(self):
        return iter(self.bound_list)
    
    def __str__(self):
        s = ''
        for i, bound in enumerate(self.bound_list):
            s += f"{i} {bound}\n"
        s += f"ranges: {self.ranges}, centers: {self.centers}, lengths{self.lengths}"
        return s
    
    def show(self, show_index: bool = False):
        """Plot the boundary and a small interior sample of the area.

        Args:
            show_index: Label exterior boundaries with their list indices.
        """
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
        X, Y = self.sampling_area([100, 100], scheme='uniform')
        plt.scatter(X.numpy(), Y.numpy(), s=0.05, color='green', alpha=0.5)
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
    @property
    def area_list(self) -> List['Area']:
        """Returns the list of Area objects in the domain."""
        return [self]

# --- Factory Functions ---

def circle(x: float, y: float, r: float) -> Area:
    """Create a circular area.

    Args:
        x: Center x coordinate.
        y: Center y coordinate.
        r: Positive radius.

    Returns:
        A circular ``Area``.

    Raises:
        ValueError: If ``r`` is not positive.
    """
    if r <= 0:
        raise ValueError("radius must be positive")

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

    return Area(
        [bound_up, bound_down],
        contains_fn=lambda X, Y: (X - x).square() + (Y - y).square() <= r**2,
        ranges={0: (x - r, x + r), 1: (y - r, y + r)},
    )

def rectangle(x_range: List[float], y_range: List[float]) -> Area:
    """Create an axis-aligned rectangular area.

    Args:
        x_range: Two-element x interval.
        y_range: Two-element y interval.

    Returns:
        A rectangular ``Area``.
    """
    p1 = [x_range[0], y_range[0]]
    p2 = [x_range[1], y_range[0]]
    p3 = [x_range[1], y_range[1]]
    p4 = [x_range[0], y_range[1]]
    return polygon(p1, p2, p3, p4)

def line_horizontal(y: float, range_x: List[float]) -> Bound:
    """Create a horizontal boundary at a fixed y coordinate."""
    return Bound(range_x, lambda x: y * torch.ones_like(x), ref_axis='x')

def line_vertical(x: float, range_y: List[float]) -> Bound:
    """Create a vertical boundary at a fixed x coordinate."""
    bound = Bound(range_y, lambda y: x * torch.ones_like(y), ref_axis='y')
    # Using a large slope to approximate verticality for x-based lookups
    bound.define_func([x - EPS, x + EPS], 
                      lambda x_: LARGE_SLOPE * (x_ - x) + (range_y[0] + range_y[1]) / 2, 
                      ref_axis='x')
    return bound

def line(pos1: List[float], pos2: List[float]) -> Bound:
    """Create a straight boundary between two points.

    Args:
        pos1: First ``[x, y]`` endpoint.
        pos2: Second ``[x, y]`` endpoint.

    Returns:
        A ``Bound`` representing the line segment.
    """
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
    """Create a polygon from ordered vertex coordinates.

    Args:
        *pos: At least three ``[x, y]`` vertices in clockwise or
            counter-clockwise order.

    Returns:
        A polygonal ``Area``.

    Raises:
        ValueError: If fewer than three vertices are supplied, vertices do not
            have shape ``(n, 2)``, or a coordinate is non-finite.
    """
    if len(pos) < 3:
        raise ValueError("polygon requires at least three vertices")

    vertices = torch.as_tensor(pos, dtype=get_dtype())
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("polygon vertices must have shape (n, 2)")
    if not torch.isfinite(vertices).all():
        raise ValueError("polygon vertices must be finite")

    bound_list = []
    for i in range(len(pos)):
        # Connect current point to the previous point
        bound_list.append(line(pos[i], pos[i-1]))

    ranges = {
        0: (vertices[:, 0].min().item(), vertices[:, 0].max().item()),
        1: (vertices[:, 1].min().item(), vertices[:, 1].max().item()),
    }
    return Area(
        bound_list,
        contains_fn=lambda x, y: _polygon_contains(vertices, x, y),
        ranges=ranges,
    )

def _polygon_contains(
    vertices: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Vectorized even-odd containment test for a simple polygon."""
    vertices = vertices.to(dtype=x.dtype, device=x.device)
    x1 = vertices[:, 0]
    y1 = vertices[:, 1]
    x2 = torch.roll(x1, shifts=-1)
    y2 = torch.roll(y1, shifts=-1)

    px = x.unsqueeze(-1)
    py = y.unsqueeze(-1)
    dy = y2 - y1
    safe_dy = torch.where(dy == 0, torch.ones_like(dy), dy)

    crosses_y = (y1 > py) != (y2 > py)
    intersection_x = x1 + (py - y1) * (x2 - x1) / safe_dy
    crossing_count = (crosses_y & (px < intersection_x)).sum(dim=-1)
    inside = crossing_count.remainder(2).bool()

    dx = x2 - x1
    cross_product = (px - x1) * dy - (py - y1) * dx
    coordinate_scale = torch.maximum(px.abs(), py.abs())
    coordinate_scale = torch.maximum(
        coordinate_scale,
        torch.maximum(
            torch.maximum(x1.abs(), x2.abs()),
            torch.maximum(y1.abs(), y2.abs()),
        ),
    )
    tolerance = torch.finfo(x.dtype).eps * 16 * (coordinate_scale + 1)
    on_segment = (
        (cross_product.abs() <= tolerance * (dx.abs() + dy.abs() + 1))
        & (px >= torch.minimum(x1, x2) - tolerance)
        & (px <= torch.maximum(x1, x2) + tolerance)
        & (py >= torch.minimum(y1, y2) - tolerance)
        & (py <= torch.maximum(y1, y2) + tolerance)
    ).any(dim=-1)
    return inside | on_segment

def curve(range_val: List[float], *func, ref_axis='x') -> Bound:
    """Create a parameterized boundary from coordinate functions.

    Args:
        range_val: Interval for the reference axis or parameter.
        *func: Boundary functions accepted by ``Bound``.
        ref_axis: Reference axis: ``"x"``, ``"y"``, or ``"t"``.

    Returns:
        A parameterized ``Bound``.
    """
    return Bound(range_val, *func, ref_axis=ref_axis)

def point(x, y) -> Bound:
    """Create a point-like boundary at ``(x, y)``."""
    return line_horizontal(y, [x-EPS, x+EPS])
