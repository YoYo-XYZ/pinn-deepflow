from xml.parsers.expat import model
from .geometry import Area, Bound
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
            bound_list+=geometry.bound_list
        else:
            raise TypeError(f"Expected Bound or Area, got {type(geometry)}")
    return ProblemDomain(bound_list, area_list)

class ProblemDomain():
    def __init__(self, bound_list:list[Bound], area_list:list[Area]):
        self.bound_list = bound_list
        self.area_list = area_list
        self.sampling_option = None
        
    def __str__(self):
        return f"""number of bound : {[f'{i}: {len(bound.X)}' for i, bound in enumerate(self.bound_list)]}\n
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
            self.bound_list[i].sampling_line(res, random=True)
            self.bound_list[i].process_coordinates()
        for i, res in enumerate(area_sampling_res):
            self.area_list[i].sampling_area(res, random=True)
            self.area_list[i].process_coordinates()

    def sampling_RAR(self, model, bound_top_k_list:list, area_top_k_list:list, bound_candidates_num_list:list=None, area_candidates_num_list:list=None):
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
                bound.X = torch.cat([original_X, X]) if original_X is not None else X
                bound.Y = torch.cat([original_Y, Y]) if original_Y is not None else Y
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
                area.X = torch.cat([original_X, X]) if original_X is not None else X
                area.Y = torch.cat([original_Y, Y]) if original_Y is not None else Y
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
        plt.figure(figsize=(10,10))
        
        self._plot_items(self.area_list, "Area", lambda o, i: (o.X, o.Y),
            {'s': 2, 'color': 'black', 'alpha': 0.3},
            {'fontsize': 20, 'color': 'navy', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_conditions)
            
        self._plot_items(self.bound_list, "Bound", lambda o, i: (o.X, o.Y),
            {'s': 2, 'color': 'red', 'alpha': 0.5},
            {'fontsize': 16, 'color': 'darkgreen', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)},
            show_label=display_conditions)
            
        plt.gca().set_aspect('equal', adjustable='box')
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.show()

    def show_setup(self, bound_sampling_res:list=None, area_sampling_res:list=None, xlim=None, ylim=None):
        plt.figure(figsize=(10,10))
        
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

        self._plot_items(self.area_list, "Area", get_area_xy,
            {'s': 5, 'color': 'lightgrey', 'alpha': 1, 'marker': 's'},
            {'fontsize': 20, 'color': 'navy', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.2, edgecolor='none', pad=1)})
        self._plot_items(self.bound_list, "Bound", get_bound_xy,
            {'s':5, 'color': 'red', 'alpha': 0.2},
            {'fontsize': 16, 'color': 'darkgreen', 'fontstyle': 'italic', 'fontweight': 'bold', 'family': 'serif', 
             'bbox': dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)})
             
        plt.gca().set_aspect('equal', adjustable='box')
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.show()

#-------------------------------------------------------------------------------------------------
    def __getitem__(self, key):
        return
    
def calc_loss_simple(domain: ProblemDomain) -> callable:
    """Returns a simple loss calculation for the given domain for PINN training."""
    def calc_loss_function(model):
        bc_loss = sum(b.calc_loss(model) for b in domain.bound_list)
        pde_loss = sum(a.calc_loss(model) for a in domain.area_list)
        total_loss = bc_loss + pde_loss
        return {"bc_loss": bc_loss, "pde_loss": pde_loss, "total_loss": total_loss}
    return calc_loss_function
