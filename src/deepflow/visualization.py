from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    area_per_plot = 50 # inches squared

    """
    Main visualization class for processing data dictionaries.
    Refactored for simplicity and maintainability.
    """
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.data_dict = data_dict
        # Cache coordinates for convenience, if they exist
        self.X = data_dict.get('x')
        self.Y = data_dict.get('y')

    def _normalize_args(self, keys: Union[str, List[str], Dict[str, str]], default_cmap: str = 'viridis') -> List[Tuple[str, str]]:
        """
        Normalizes input keys to a list of (key, colormap) tuples.
        Supports str, list of str, or dict {key: cmap}.
        Filters out keys that are not present in data_dict.
        """
        if isinstance(keys, str):
            items = [(keys, default_cmap)]
        elif isinstance(keys, (list, tuple)):
            items = [(k, default_cmap) for k in keys]
        elif isinstance(keys, dict):
            items = [(k, v if v is not None else default_cmap) for k, v in keys.items()]
        else:
            raise TypeError("Input must be a string, list, or dictionary.")
            
        # Filter keys
        valid_items = []
        for k, cmap in items:
            if k in self.data_dict:
                valid_items.append((k, cmap))
            else:
                print(f"Warning: Key '{k}' not found in data_dict. Skipping.")
        
        if len(valid_items) == 0:
            print("Warning: No valid keys found to plot.")
        
        return valid_items

    def _create_subplots(self, n_plots: int, orientation: str = 'vertical', subplot_kw: Optional[Dict] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Creates a figure and a list of axes based on the number of plots and orientation.
        """
        # Determine the ratio of layout based from X and Y ranges
        try: ratio = np.ptp(self.Y) / np.ptp(self.X)
        except Exception: pass
        if ratio == 0: ratio = 0.4/np.ptp(self.X)
        elif ratio > 1e5: ratio = 0.4*np.ptp(self.Y)

        #Determine size of the plots based from ration and area per plot
        rows, cols = (n_plots, 1) if orientation == 'vertical' else (1, n_plots)
        fig_length, fig_width = (self.area_per_plot / ratio)**0.5, (self.area_per_plot * ratio)**0.5 #parameterized width and length to area and ratio
        if orientation == 'vertical':
            figsize = (fig_length, fig_width * n_plots)
        else:
            figsize = (fig_length * n_plots, fig_width)

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True, subplot_kw=subplot_kw)
        
        # Flatten axes array for easy iteration
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        return fig, axes


    def plot_color(self, keys: Union[str, List[str], Dict], s: Union[int, float] = 2, orientation: str = 'vertical') -> plt.Figure:
        """
        Creates scatter plots (heatmap style) for the specified keys.
        """
        if self.X is None or self.Y is None:
            raise ValueError("Data dictionary must contain 'x' and 'y' for scatter plots.")

        items = self._normalize_args(keys)

        fig, axes = self._create_subplots(len(items), orientation)

        for ax, (key, cmap) in zip(axes, items):
            # Plot
            im = ax.scatter(self.X, self.Y, s=s, c=self.data_dict[key], cmap=cmap, marker='s')
            
            # Styling
            ax.set_title(key, fontweight='medium', pad=10, fontsize=13)    
            ax.set_xlabel('x', fontstyle='italic', labelpad=0)
            ax.set_ylabel('y', fontstyle='italic', labelpad=0)
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax, pad=0.03)

        return fig
    
    # Modern alias
    plot_scatter = plot_color

    def plot(self, keys: Union[str, List[str], Dict], axis: str = 'xy') -> plt.Figure:
        """
        General plotting method.
        If axis='xy': 3D surface plot.
        If axis='x' or 'y': 1D line plot against that axis.
        """
        items = self._normalize_args(keys, default_cmap='viridis' if axis == 'xy' else None)
        
        is_3d = (axis == 'xy')
        subplot_kw = {'projection': '3d'} if is_3d else {}
        
        # Default orientation vertical for consistency with old behavior
        fig, axes = self._create_subplots(len(items), 'vertical', subplot_kw=subplot_kw)

        for ax, (key, color) in zip(axes, items):
            if is_3d:
                # 3D Scatter Plot
                scatter = ax.scatter(self.X, self.Y, self.data_dict[key], c=self.data_dict[key], cmap=color, s=2)
                ax.set(xlabel='x', ylabel='y')
                ax.set_title(f'3D Scatter Plot of {key}')
                ax.dist = 8 # Compact view
                fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
            
            elif axis in ['x', 'y']:
                # Line Plot
                ax.plot(self.data_dict[axis], self.data_dict[key], linewidth=2.0, color=color)
                ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
                ax.set_xlabel(axis, fontsize=10)
                ax.set_ylabel(key, fontsize=10)
            
            else:
                raise ValueError("axis must be 'x', 'y', or 'xy'")

        return fig

    def plot_distribution(self, keys: Union[str, List[str]], bins: Union[str, int] = 'fd') -> plt.Figure:
        """
        Plots histograms for the specified keys.
        """
        if isinstance(keys, str): keys = [keys]
        
        fig, axes = self._create_subplots(len(keys), 'vertical')

        for ax, key in zip(axes, keys):
            if key in self.data_dict:
                ax.hist(self.data_dict[key], bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="black")
                ax.set_title(f"{key} distribution")
        
        return fig

    def plot_loss_curve(self, log_scale: bool = False, linewidth: float = 0.5, 
                        start: int = 0, end: Optional[int] = None, 
                        keys: List[str] = ['total_loss', 'bc_loss', 'pde_loss']) -> plt.Figure:
        """
        Plots loss per iteration.
        """
        fig, ax = plt.subplots()

        for key in keys:
            if key in self.data_dict:
                ax.plot(self.data_dict[key][start:end], label=key, linewidth=linewidth)

        if log_scale:
            ax.set_yscale("log")
            
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per Iteration")
        ax.legend()  
        
        return fig


    
