"""Problem-domain orchestration, sampling, and PINN loss construction."""

from typing import TYPE_CHECKING

from .geometry import Area, Bound
try:
    import ultraplot as plt
except ImportError:
    import matplotlib.pyplot as plt
import torch
import sympy as sp
from .nn import HardConstraint
from .geometry import CustomData

if TYPE_CHECKING:
    from .evaluation import ReferenceGroupEvaluator


def domain(*geometries):
    """Build a ``ProblemDomain`` from bounds, areas, or custom data.

    Lists must contain only bounds or only areas. Unsupported inputs raise
    ``TypeError``.
    """
    bound_list = []
    area_list = []
    for geometry in geometries:
        if isinstance(geometry, list):
            if all(isinstance(g, Bound) for g in geometry):
                bound_list+=geometry
            elif all(isinstance(g, Area) for g in geometry):
                area_list+=geometry
            else:
                raise TypeError("List must contain only Bound or Area objects")
        elif isinstance(geometry, Bound):
            bound_list.append(geometry)
        elif isinstance(geometry, Area):
            area_list.append(geometry)
            bound_list += geometry.bound_list
        elif isinstance(geometry, CustomData):
            area_list.append(geometry)
        else:
            raise TypeError(f"Expected Bound or Area, got {type(geometry)}")
    return ProblemDomain(bound_list, area_list)


class ProblemDomain():
    """Group boundaries and areas for sampling, training, and evaluation.

    Args:
        bound_list: Boundaries, including area boundaries.
        area_list: Areas or custom data.

    Sampling updates coordinates in place. Use ``geometry.evaluate(model)`` for
    results; ``sampling_option`` records the last sampling strategy.
    """

    def __init__(self, bound_list:list[Bound], area_list:list[Area]):
        """Initialize a domain from its boundary and area lists."""
        self.bound_list = bound_list
        self.area_list = area_list
        self.sampling_option = None
        
        for g in self.bound_list + self.area_list:
            if isinstance(g, CustomData):
                g.process_coordinates()

    def solve_fem(
        self,
        mesh_size=None,
        boundary_resolution=128,
        time_step=None,
        tolerance=1e-8,
        max_iterations=200,
    ) -> "ReferenceGroupEvaluator":
        """Solve with optional NGSolve FEM and return a
        ``ReferenceGroupEvaluator``.

        Args:
            mesh_size: Target mesh size; ``None`` infers it from the area.
            boundary_resolution: Samples per mesh boundary.
            time_step: Transient snapshot spacing.
            tolerance: Solver convergence tolerance.
            max_iterations: Solver iteration limit.

        Access the solution through ``.reference_solution``. Missing NGSolve
        raises ``ImportError``; invalid domains or settings raise
        ``ValueError``.
        """
        from .fem import solve_fem as _solve_fem

        return _solve_fem(
            self,
            mesh_size=mesh_size,
            boundary_resolution=boundary_resolution,
            time_step=time_step,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    def __str__(self):
        return f"""number of bound : {[f'{i}: {len(bound.X)}' for i, bound in enumerate(self.bound_list)]}
number of area : {[f'{i}: {len(area.X)}' for i, area in enumerate(self.area_list)]}"""

    def sampling_uniform(self, bound_sampling_res:list=[], area_sampling_res:list=[]):
        """Replace geometry coordinates with uniform-grid samples.

        Args:
            bound_sampling_res: Resolution per boundary.
            area_sampling_res: Resolution per area.
        """
        self.sampling_option = 'uniform'
        for i, res in enumerate(bound_sampling_res):
            self.bound_list[i].sampling_line(res)
            self.bound_list[i].process_coordinates()
        for i, res in enumerate(area_sampling_res):
            self.area_list[i].sampling_area(res)
            self.area_list[i].process_coordinates()

    def sampling_random(self, bound_sampling_res:list=[], area_sampling_res:list=[]):
        """Replace geometry coordinates with uniform random samples.

        Args:
            bound_sampling_res: Point count per boundary.
            area_sampling_res: Candidate count per area.
        """
        self.sampling_option = 'random'
        for i, res in enumerate(bound_sampling_res):
            self.bound_list[i].sampling_line(res, scheme='random')
            self.bound_list[i].process_coordinates()
        for i, res in enumerate(area_sampling_res):
            self.area_list[i].sampling_area(res, scheme='random')
            self.area_list[i].process_coordinates()

    def sampling_lhs(self, bound_sampling_res:list=[], area_sampling_res:list=[]):
        """Replace geometry coordinates with Latin Hypercube samples.

        Args:
            bound_sampling_res: Point count per boundary.
            area_sampling_res: Candidate count per area.
        """
        self.sampling_option = 'lhs'
        for i, res in enumerate(bound_sampling_res):
            self.bound_list[i].sampling_line(res, scheme='lhs')
            self.bound_list[i].process_coordinates()
        for i, res in enumerate(area_sampling_res):
            self.area_list[i].sampling_area(res, scheme='lhs')
            self.area_list[i].process_coordinates()

    def sampling_RAR(self, bound_top_k_list:list=None, area_top_k_list:list=None, bound_candidates_num_list:list=None, area_candidates_num_list:list=None):
        """Append high-residual points to existing samples. Compute residuals
        first.

        Args:
            bound_top_k_list: Points to retain per boundary.
            area_top_k_list: Points to retain per area.
            bound_candidates_num_list: Candidates per boundary.
            area_candidates_num_list: Candidates per area.
        """
        self.sampling_option = self.sampling_option + ' + RAR'
        if bound_top_k_list:
            for i, bound in enumerate(self.bound_list):
                bound.save_coordinates()
                bound.sampling_line(bound_candidates_num_list[i], scheme='lhs')
                bound.process_coordinates()
                bound.get_residual_based_points_topk(top_k=bound_top_k_list[i])
                bound.apply_residual_based_points()
                bound.clear_residual_based_points()
                bound.process_coordinates()
        if area_top_k_list:
            for i, area in enumerate(self.area_list):
                area.save_coordinates()
                area.sampling_area(area_candidates_num_list[i], scheme='lhs')
                area.process_coordinates()
                area.get_residual_based_points_topk(top_k=area_top_k_list[i])
                area.apply_residual_based_points()
                area.process_coordinates()
                area.clear_residual_based_points()

    def sampling_R3(self, bound_sampling_res:list=None, area_sampling_res:list=None):
        """Resample with a fixed point budget per geometry.

        Keep points above the mean residual and fill with random samples. If
        too many qualify, keep the highest-residual points up to the budget.

        Args:
            bound_sampling_res: Point budget per boundary.
            area_sampling_res: Point budget per area.
        """
        self.sampling_option = self.sampling_option + ' + R3'

        def resample_geometry(geometry, res, sampler):
            geometry.get_residual_based_points_threshold()
            if len(geometry.X_residual_container[0]) > res:
                _, top_indices = torch.topk(geometry.residual_field, res)
                top_indices = top_indices.cpu()
                geometry.X_residual_container = [geometry.X[top_indices]]
                geometry.Y_residual_container = [geometry.Y[top_indices]]

            remaining = res - len(geometry.X_residual_container[0])
            if remaining:
                sampler(remaining, scheme='random')
            else:
                geometry.X = geometry.X[:0]
                geometry.Y = geometry.Y[:0]
            geometry.apply_residual_based_points()
            geometry.process_coordinates()

        if bound_sampling_res:
            for i, res in enumerate(bound_sampling_res):
                resample_geometry(
                    self.bound_list[i], res, self.bound_list[i].sampling_line
                )
        if area_sampling_res:
            for i, res in enumerate(area_sampling_res):
                resample_geometry(
                    self.area_list[i], res, self.area_list[i].sampling_area
                )

    def sampling_R3_(self, bound_sampling_res:list=None, area_sampling_res:list=None):
        """Add fresh random samples to points above the mean residual.

        Point counts may grow on each call.

        Args:
            bound_sampling_res: Fresh candidates per boundary.
            area_sampling_res: Fresh candidates per area.
        """
        self.sampling_option = self.sampling_option + ' + R3'
        
        if bound_sampling_res:
            for i, res in enumerate(bound_sampling_res):
                self.bound_list[i].get_residual_based_points_threshold()
                self.bound_list[i].sampling_line(res, scheme='random')
                self.bound_list[i].apply_residual_based_points()
                self.bound_list[i].process_coordinates()
        if area_sampling_res:
            for i, res in enumerate(area_sampling_res):
                self.area_list[i].get_residual_based_points_threshold()
                self.area_list[i].sampling_area(res, scheme='random')
                self.area_list[i].apply_residual_based_points()
                self.area_list[i].process_coordinates()

    def sampling_accumulate(self, bound_sampling_res:list=None, area_sampling_res:list=None):
        """Grow the training set by retaining residual-based points across calls.

        Args:
            bound_sampling_res: Fresh points per boundary.
            area_sampling_res: Fresh points per area.

        This method is experimental.
        """
        self.sampling_option = self.sampling_option + ' + R3'
        
        if bound_sampling_res:
            for i, res in enumerate(bound_sampling_res):
                self.bound_list[i].get_residual_based_points_threshold(maintain_points=True)
                self.bound_list[i].sampling_line(res, scheme='random')
                self.bound_list[i].apply_residual_based_points()
                self.bound_list[i].process_coordinates()
        if area_sampling_res:
            for i, res in enumerate(area_sampling_res):
                self.area_list[i].get_residual_based_points_threshold(maintain_points=True)
                self.area_list[i].sampling_area(res, scheme='random')
                self.area_list[i].apply_residual_based_points()
                self.area_list[i].process_coordinates()
    def clear_residual_points(self):
        """Discard residual-based point buffers from all geometries."""
        for geometry in self.bound_list + self.area_list:
            geometry.clear_residual_based_points()
    def _format_condition_dict(self, obj, obj_type='Bound'):

        def func_to_latex(func_list):
            v = func_list
            try:
                return f"${str(sp.latex(v[1](sp.symbols(v[0]))))}$"
            except Exception:
                return f"Function({v[0]})"

        if hasattr(obj, 'condition_dict') and obj.condition_dict is not None:
            conditions = f'{obj.physics_type}: ' + ', '.join([f"{k}={(str(v) if isinstance(v,(float,int,HardConstraint)) else func_to_latex(v))}" for k, v in obj.condition_dict.items()])
            return conditions
        elif hasattr(obj, 'PDE') and obj.PDE is not None:
            return f'{obj.physics_type}: ' + f'{obj.PDE.__class__.__name__}'
        return ""
    
    def save_coordinates(self):
        """Save a copy of every geometry's current coordinates."""
        for area in self.area_list:
            area.saved_X = area.X.clone()
            area.saved_Y = area.Y.clone()
        for bound in self.bound_list:
            bound.saved_X = bound.X.clone()
            bound.saved_Y = bound.Y.clone()
    
    def load_coordinates(self):
        """Restore coordinates previously saved by ``save_coordinates``."""
        for area in self.area_list:
            area.X = area.saved_X.clone()
            area.Y = area.saved_Y.clone()
        for bound in self.bound_list:
            bound.X = bound.saved_X.clone()
            bound.Y = bound.saved_Y.clone()
    
    def _plot_items(self, ax, items, name, get_xy, scatter_kw, text_kw, show_label=True):
        for i, obj in enumerate(items):
            x, y = get_xy(obj, i)
            if hasattr(x, 'detach'): x = x.detach().cpu().numpy()
            if hasattr(y, 'detach'): y = y.detach().cpu().numpy()
            ax.scatter(x, y, **scatter_kw)
            if show_label:
                cond = self._format_condition_dict(obj, name)
                lbl = f"{name} {i}\n{cond}" if cond else f"{name} {i}"
                ax.text(obj.centers[0], obj.centers[1], lbl, ha='center', va='center', **text_kw)
    def show_coordinates(self, display_physics = False, xlim=None, ylim=None, display_resampling=False):
        """Plot sampled coordinates.

        Args:
            display_physics: Annotate BC, IC, and PDE assignments.
            xlim: x-axis limits.
            ylim: y-axis limits.
            display_resampling: Show buffered residual-based points.
        """
        fig, ax = plt.subplots(refwidth=7)
        
        self._plot_items(ax, self.area_list, "Area", lambda o, i: (o.X, o.Y),
            {'s': 2, 'color': 'black', 'alpha': 0.3},
            {'fontsize': 10, 'color': 'navy', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_physics)
            
        self._plot_items(ax, self.bound_list, "Bound", lambda o, i: (o.X, o.Y),
            {'s': 1, 'color': 'red', 'alpha': 0.5},
            {'fontsize': 10, 'color': 'darkgreen', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_physics)
        
        if display_resampling:
            for area in self.area_list:
                if area.X_residual_container:
                    ax.scatter(torch.cat(area.X_residual_container), torch.cat(area.Y_residual_container), s=2, color='orange', alpha=0.7)
            for bound in self.bound_list:
                if bound.X_residual_container:
                    ax.scatter(torch.cat(bound.X_residual_container), torch.cat(bound.Y_residual_container), s=1, color='green', alpha=0.7)
            ax.legend(loc='upper right')
            
        ax.set_aspect('equal', adjustable='box')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        plt.show()

    def show_setup(self, bound_sampling_res:list=None, area_sampling_res:list=None, xlim=None, ylim=None):
        """Plot geometry with temporary preview samples.

        Args:
            bound_sampling_res: Preview resolution per boundary.
            area_sampling_res: Preview resolution per area.
            xlim: x-axis limits.
            ylim: y-axis limits.
        """
        fig, ax = plt.subplots(refwidth=7, grid = False)
        
        if bound_sampling_res is None:
            bound_sampling_res = [int(800*(b.ranges[b.ax][1] - b.ranges[b.ax][0])) for b in self.bound_list]
        if area_sampling_res is None:
            area_sampling_res = [[400, int(400*a.lengths[1]/a.lengths[0])] for a in self.area_list]

        def get_area_xy(area, i):
            if isinstance(area, Area):
                area.sampling_area(area_sampling_res[i])
            return area.X, area.Y
        
        def get_bound_xy(bound, i):
            if isinstance(bound, Bound):
                bound.sampling_line(bound_sampling_res[i])
            return bound.X, bound.Y

        self._plot_items(ax, self.area_list, "Area", get_area_xy,
            {'s': 5, 'color': 'lightgrey', 'alpha': 1, 'marker': 's'},
            {'fontsize': 10, 'color': 'navy', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.2, edgecolor='none', pad=1)})
        self._plot_items(ax, self.bound_list, "Bound", get_bound_xy,
            {'s':5, 'color': 'red', 'alpha': 0.2},
            {'fontsize': 10, 'color': 'darkgreen', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)})
             
        ax.set_aspect('equal', adjustable='box')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        plt.show()

    def __getitem__(self, key):
        return
    
    def __iter__(self):
        for geometry in self.bound_list + self.area_list:
            yield geometry

    def _batched_loss(self, model) -> dict:
        loss_dict = {"pde_loss": 0.0, "bc_loss": 0.0, "ic_loss": 0.0}

        if not hasattr(self, '_groups_cache'):
            groups: dict[str, list] = {}
            for geometry in self:
                groups.setdefault(geometry.physics_type, []).append(geometry)
            self._groups_cache = groups
        groups = self._groups_cache

        for physics_type, geometries in groups.items():
            input_keys = list(geometries[0].inputs_tensor_dict.keys())
            batched_inputs = {
                k: torch.cat([g.inputs_tensor_dict[k] for g in geometries])
                for k in input_keys
            }

            batched_outputs = model(batched_inputs)

            start = 0
            for g in geometries:
                end = start + len(g.X_)
                g.model_inputs = g.inputs_tensor_dict
                g.model_outputs = {k: batched_outputs[k][start:end] for k in batched_outputs}
                g._compute_residual_field()
                loss_dict[f'{physics_type.lower()}_loss'] += torch.mean(
                    g.residual_field_raw.square().sum(dim=0)
                )
                start = end

        return loss_dict

def calc_loss_simple(domain: ProblemDomain) -> callable:
    """Return ``loss(model)`` with ``bc_loss``, ``ic_loss``, ``pde_loss``, and
    their unweighted sum, ``total_loss``, for ``domain``.
    """
    def calc_loss_function(model):
        loss_dict = domain._batched_loss(model)
        loss_dict["total_loss"] = sum(value for key, value in loss_dict.items() if key != "total_loss")
        return loss_dict
    
    return calc_loss_function

def calc_loss_weighted(domain: ProblemDomain, bc_weights = 1, ic_weights = 1, pde_weights = 1) -> callable:
    """Return ``loss(model)`` for ``domain`` with a weighted ``total_loss``.

    Args:
        domain: Domain to evaluate.
        bc_weights: Boundary-loss weight.
        ic_weights: Initial-condition-loss weight.
        pde_weights: PDE-loss weight.
    """
    weight = {"pde_loss": pde_weights, "bc_loss": bc_weights, "ic_loss": ic_weights}
    def calc_loss_function(model):
        loss_dict = domain._batched_loss(model)
        loss_dict["total_loss"] = sum(weight[key] * value for key, value in loss_dict.items() if key != "total_loss")
        return loss_dict
    
    return calc_loss_function
