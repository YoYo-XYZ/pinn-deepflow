from .geometry import Area, Bound
try:
    import ultraplot as plt
except ImportError:
    import matplotlib.pyplot as plt
import torch
import sympy as sp
from .neuralnetwork import HardConstraint

def domain(*geometries):
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
        else:
            raise TypeError(f"Expected Bound or Area, got {type(geometry)}")
    return ProblemDomain(bound_list, area_list)

class ProblemDomain():
    def __init__(self, bound_list:list[Bound], area_list:list[Area]):
        self.bound_list = bound_list
        self.area_list = area_list
        self.sampling_option = None
        
    def __str__(self):
        return f"""number of bound : {[f'{i}: {len(bound.X)}' for i, bound in enumerate(self.bound_list)]}
number of area : {[f'{i}: {len(area.X)}' for i, area in enumerate(self.area_list)]}"""

    def sampling_uniform(self, bound_sampling_res:list=[], area_sampling_res:list=[]):
        self.sampling_option = 'uniform'
        for i, res in enumerate(bound_sampling_res):
            self.bound_list[i].sampling_line(res)
            self.bound_list[i].process_coordinates()
        for i, res in enumerate(area_sampling_res):
            self.area_list[i].sampling_area(res)
            self.area_list[i].process_coordinates()

    def sampling_random(self, bound_sampling_res:list=[], area_sampling_res:list=[]):
        self.sampling_option = 'random'
        for i, res in enumerate(bound_sampling_res):
            self.bound_list[i].sampling_line(res, scheme='random')
            self.bound_list[i].process_coordinates()
        for i, res in enumerate(area_sampling_res):
            self.area_list[i].sampling_area(res, scheme='random')
            self.area_list[i].process_coordinates()

    def sampling_lhs(self, bound_sampling_res:list=[], area_sampling_res:list=[]):
        self.sampling_option = 'lhs'
        for i, res in enumerate(bound_sampling_res):
            self.bound_list[i].sampling_line(res, scheme='lhs')
            self.bound_list[i].process_coordinates()
        for i, res in enumerate(area_sampling_res):
            self.area_list[i].sampling_area(res, scheme='lhs')
            self.area_list[i].process_coordinates()

    def sampling_RAR(self, bound_top_k_list:list=None, area_top_k_list:list=None, bound_candidates_num_list:list=None, area_candidates_num_list:list=None):
        self.sampling_option = self.sampling_option + ' + RAR'
        if bound_top_k_list:
            for i, bound in enumerate(self.bound_list):
                bound.save_coordinates()
                # Sample new candidates
                bound.sampling_line(bound_candidates_num_list[i], scheme='lhs')
                bound.process_coordinates()
                # Add RAR point to saved points
                bound.get_residual_based_points_topk(top_k=bound_top_k_list[i])
                bound.apply_residual_based_points()
                bound.clear_residual_based_points()
                bound.process_coordinates()
        if area_top_k_list:
            for i, area in enumerate(self.area_list):
                area.save_coordinates()
                # Sample new candidates
                area.sampling_area(area_candidates_num_list[i], scheme='lhs')
                area.process_coordinates()
                # Add RAR point to saved points
                area.get_residual_based_points_topk(top_k=area_top_k_list[i])
                area.apply_residual_based_points()
                area.process_coordinates()
                area.clear_residual_based_points()

    def sampling_R3(self, bound_sampling_res:list=None, area_sampling_res:list=None):
        self.sampling_option = self.sampling_option + ' + R3'
        if bound_sampling_res:
            for i, bound in enumerate(self.bound_list):
                # Sample new candidates
                bound.get_residual_based_points_threshold()
                bound.sampling_line(bound_sampling_res[i], scheme='lhs')
                bound.apply_residual_based_points()
                bound.process_coordinates()
                # Add RAR point to saved points
        if area_sampling_res:
            for i, area in enumerate(self.area_list):
                # Sample new candidates
                area.get_residual_based_points_threshold()
                area.sampling_area(area_sampling_res[i], scheme='lhs')
                area.apply_residual_based_points()
                area.process_coordinates()
                # Add RAR point to saved points

#------------------------------------------------------------------------------------------------
    def _format_condition_dict(self, obj, obj_type='Bound'):
        """Helper function to format condition dictionary for display."""

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
        for area in self.area_list:
            area.saved_X = area.X.clone()
            area.saved_Y = area.Y.clone()
        for bound in self.bound_list:
            bound.saved_X = bound.X.clone()
            bound.saved_Y = bound.Y.clone()
    
    def load_coordinates(self):
        for area in self.area_list:
            area.X = area.saved_X.clone()
            area.Y = area.saved_Y.clone()
        for bound in self.bound_list:
            bound.X = bound.saved_X.clone()
            bound.Y = bound.saved_Y.clone()
    
#-------------------------------------------------------------------------------------------------
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
    def show_coordinates(self, display_physics = False, xlim=None, ylim=None):
        fig, ax = plt.subplots(refwidth=7)
        
        self._plot_items(ax, self.area_list, "Area", lambda o, i: (o.X, o.Y),
            {'s': 2, 'color': 'black', 'alpha': 0.3},
            {'fontsize': 10, 'color': 'navy', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_physics)
            
        self._plot_items(ax, self.bound_list, "Bound", lambda o, i: (o.X, o.Y),
            {'s': 2, 'color': 'red', 'alpha': 0.5},
            {'fontsize': 10, 'color': 'darkgreen', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_physics)
            
        ax.set_aspect('equal', adjustable='box')
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        plt.show()

    def show_setup(self, bound_sampling_res:list=None, area_sampling_res:list=None, xlim=None, ylim=None):
        fig, ax = plt.subplots(refwidth=7, grid = False)
        
        if bound_sampling_res is None:
            bound_sampling_res = [int(800*(b.ranges[b.ax][1] - b.ranges[b.ax][0])) for b in self.bound_list]
        if area_sampling_res is None:
            area_sampling_res = [[400, int(400*a.lengths[1]/a.lengths[0])] for a in self.area_list]

        def get_area_xy(area, i):
            area.sampling_area(area_sampling_res[i])
            return area.X, area.Y
        
        def get_bound_xy(bound, i):
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

#-------------------------------------------------------------------------------------------------
    def __getitem__(self, key):
        return
    
    def __iter__(self):
        for geometry in self.bound_list + self.area_list:
            yield geometry
    
def calc_loss_simple(domain: ProblemDomain) -> callable:
    """Returns a simple loss calculation for the given domain for PINN training."""
    def calc_loss_function(model):
        loss_dict = {"pde_loss": 0.0, "bc_loss": 0.0, "ic_loss": 0.0}

        for geometry in domain:
                try:
                    loss_dict[f'{geometry.physics_type.lower()}_loss'] += geometry.calc_loss(model)
                except Exception:
                    pass
        loss_dict["total_loss"] = sum(value for key, value in loss_dict.items() if key != "total_loss")
        return loss_dict
    
    return calc_loss_function
