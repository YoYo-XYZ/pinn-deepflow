from .Geometry import Area, Bound
import matplotlib.pyplot as plt
import torch
import sympy as sp
from .Network import HardConstraint

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
        else:
            raise TypeError(f"Expected Bound or Area, got {type(geometry)}")
    return ProblemDomain(bound_list, area_list)

class ProblemDomain():
    def __init__(self, bound_list, area_list):
        self.bound_list = bound_list
        self.area_list = area_list
        self.sampling_option = None
        
    def __str__(self):
        return f"""number of bound : {len(self.bound_list)}
        {[f'{i}: {len(bound.X)}' for i, bound in enumerate(self.bound_list)]}
        , number of area : {len(self.area_list)}
        {[f'{i}: {len(area.X)}' for i, area in enumerate(self.area_list)]}"""

    def sampling_uniform(self, bound_sampling_res:list, area_sampling_res:list):
        self.sampling_option = 'uniform'
        for i, bound in enumerate(self.bound_list):
            if bound_sampling_res[i] is not None:
                bound.sampling_line(bound_sampling_res[i])
                bound.process_coordinates()
        for i, area in enumerate(self.area_list):
            if area_sampling_res[i] is not None:
                area.sampling_area(area_sampling_res[i])
                area.process_coordinates()

    def sampling_random_r(self, bound_sampling_res:list, area_sampling_res:list):
        self.sampling_option = 'random_r'
        for i, bound in enumerate(self.bound_list):
            if bound_sampling_res[i] is not None:
                bound.sampling_line(bound_sampling_res[i], random=True)
                bound.process_coordinates()
        for i, area in enumerate(self.area_list):
            if area_sampling_res[i] is not None:
                area.sampling_area(area_sampling_res[i], random=True)
                area.process_coordinates()

    def sampling_RAR(self, bound_top_k_list:list, area_top_k_list:list, model, bound_candidates_num_list:list=None, area_candidates_num_list:list=None):
        self.sampling_option = self.sampling_option + ' + RAR'
        for i, bound in enumerate(self.bound_list):
            if bound_candidates_num_list is None:
                bound.sampling_residual_based(bound_top_k_list[i], model)
            else:
                # Create a temporary copy by saving current state
                original_X = bound.X.clone() if hasattr(bound, 'X') else None
                original_Y = bound.Y.clone() if hasattr(bound, 'Y') else None
                
                # Sample new candidates
                bound.sampling_line(bound_candidates_num_list[i], random=True)
                bound.process_coordinates()
                X, Y = bound.sampling_residual_based(bound_top_k_list[i], model)
                
                # Restore and concatenate
                if original_X is not None:
                    bound.X = torch.cat([original_X, X])
                    bound.Y = torch.cat([original_Y, Y])
                else:
                    bound.X = X
                    bound.Y = Y
            bound.process_coordinates()
        for i, area in enumerate(self.area_list):
            if area_candidates_num_list is None:
                area.sampling_residual_based(area_top_k_list[i], model)
            else:
                # Create a temporary copy by saving current state
                original_X = area.X.clone() if hasattr(area, 'X') else None
                original_Y = area.Y.clone() if hasattr(area, 'Y') else None
                
                # Sample new candidates
                area.sampling_area(area_candidates_num_list[i], random=True)
                area.process_coordinates()
                X, Y = area.sampling_residual_based(area_top_k_list[i], model)
                
                # Restore and concatenate
                if original_X is not None:
                    area.X = torch.cat([original_X, X])
                    area.Y = torch.cat([original_Y, Y])
                else:
                    area.X = X
                    area.Y = Y
            area.process_coordinates()
#------------------------------------------------------------------------------------------------
    def _format_condition_dict(self, obj, obj_type='Bound'):
        """Helper function to format condition dictionary for display."""

        def func_to_latex(func_list):
            v = func_list
            return f"${str(sp.latex(v[1](sp.symbols(v[0]))))}$"

        if hasattr(obj, 'condition_dict'):
            conditions = ', '.join([f"{k}={(str(v) if isinstance(v,(float,int,HardConstraint)) else func_to_latex(v))}" for k, v in obj.condition_dict.items()])
            return conditions
        elif hasattr(obj, 'PDE'):
            return f"PDE: {obj.PDE.__class__.__name__}"
        return ""
    
    def save_coordinates(self):
        for i, area in enumerate(self.area_list):
            area.saved_X = area.X.clone()
            area.saved_Y = area.Y.clone()
        for i, bound in enumerate(self.bound_list):
            bound.saved_X = bound.X.clone()
            bound.saved_Y = bound.Y.clone()
    
    def load_coordinates(self):
        for i, area in enumerate(self.area_list):
            area.X = area.saved_X.clone()
            area.Y = area.saved_Y.clone()
        for i, bound in enumerate(self.bound_list):
            bound.X = bound.saved_X.clone()
            bound.Y = bound.saved_Y.clone()
    
#-------------------------------------------------------------------------------------------------
    def _plot_items(self, items, name, get_xy, scatter_kw, text_kw, show_label=True):
        for i, obj in enumerate(items):
            x, y = get_xy(obj, i)
            if hasattr(x, 'detach'): x = x.detach().cpu().numpy()
            if hasattr(y, 'detach'): y = y.detach().cpu().numpy()
            plt.scatter(x, y, **scatter_kw)
            if show_label:
                cond = self._format_condition_dict(obj, name)
                lbl = f"{name} {i}\n{cond}" if cond else f"{name} {i}"
                plt.text(obj.centers[0], obj.centers[1], lbl, ha='center', va='center', **text_kw)

    def show_coordinates(self, display_conditions=False, xlim=None, ylim=None):
        plt.figure(figsize=(20,20))
        
        self._plot_items(self.area_list, "Area", lambda o, i: (o.X, o.Y),
            {'s': 5, 'color': 'black', 'alpha': 0.3},
            {'fontsize': 20, 'color': 'navy', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_conditions)
            
        self._plot_items(self.bound_list, "Bound", lambda o, i: (o.X, o.Y),
            {'s': 10, 'color': 'red', 'alpha': 0.5},
            {'fontsize': 16, 'color': 'darkgreen', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_conditions)
            
        plt.gca().set_aspect('equal', adjustable='box')
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.show()

    def show_setup(self, bound_sampling_res:list=None, area_sampling_res:list=None, xlim=None, ylim=None):
        plt.figure(figsize=(20,20))
        
        if bound_sampling_res is None:
            bound_sampling_res = [int(800*(b.ranges[b.ax][1] - b.ranges[b.ax][0])) for b in self.bound_list]
        if area_sampling_res is None:
            area_sampling_res = [[200, int(200*a.lengths[1]/a.lengths[0])] for a in self.area_list]

        def get_area_xy(area, i):
            area.sampling_area(area_sampling_res[i])
            return area.X, area.Y
        
        def get_bound_xy(bound, i):
            bound.sampling_line(bound_sampling_res[i])
            return bound.X, bound.Y

        self._plot_items(self.area_list, "Area", get_area_xy,
            {'s': 1, 'color': 'black', 'alpha': 0.5, 'marker': 's'},
            {'fontsize': 20, 'color': 'navy', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.2, edgecolor='none', pad=1)})
        self._plot_items(self.bound_list, "Bound", get_bound_xy,
            {'s': 5, 'color': 'red', 'alpha': 0.5},
            {'fontsize': 16, 'color': 'darkgreen', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)})
             
        plt.gca().set_aspect('equal', adjustable='box')
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.show()