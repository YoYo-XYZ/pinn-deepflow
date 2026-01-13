from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np

class VisualizeTemplate:
    """
    Base class containing static methods for specific plot types.
    """
    
    @staticmethod
    def scatterplot(x: np.ndarray, y: np.ndarray, data: np.ndarray, 
                  ax: matplotlib.axes.Axes, title: Optional[str] = None, 
                  cmap: str = 'viridis', s: int = 1) -> matplotlib.axes.Axes:
        """Generates a scatter plot with color mapping."""
        im = ax.scatter(x, y, s=s, c=data, cmap=cmap, marker='s')
        
        if title:
            ax.set_title(title, fontweight='medium', pad=10, fontsize=13)
            
        ax.set_xlabel('x', fontstyle='italic', labelpad=0)
        ax.set_ylabel('y', fontstyle='italic', labelpad=0)

        # Access the figure from the axes to add colorbar
        if ax.figure:
            ax.figure.colorbar(im, ax=ax, pad=0.03, shrink=1.2)

        ax.set_aspect('equal', adjustable='box')
        return ax

    @staticmethod
    def lineplot(x: np.ndarray, data: np.ndarray, ax: matplotlib.axes.Axes, 
                 xlabel: str, ylabel: str, color: str = 'navy') -> matplotlib.axes.Axes:
        """Generates a standard line plot."""
        ax.plot(x, data, linewidth=2.0, color=color)
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        return ax

    @staticmethod
    def histplot(data: np.ndarray, ax: matplotlib.axes.Axes, 
                 title: Optional[str] =  None, bins: Union[str, int] = 'fd') -> matplotlib.axes.Axes:
        """Generates a histogram."""
        ax.hist(data, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="black")
        if title:
            ax.set_title(title)
        return ax

    @staticmethod
    def surfaceplot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                     ax: matplotlib.axes.Axes, title: Optional[str] = None, 
                     cmap: str = 'viridis') -> matplotlib.axes.Axes:
        """Generates a 3D surface plot."""
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
        if title:
            ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.figure.colorbar(surf, ax=ax, pad=0.1)
        return ax

class Visualizer():
    """
    Main visualization class for processing data dictionaries.
    """
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        # Error handling: Ensure required keys exist
        if 'x' not in data_dict or 'y' not in data_dict:
            raise KeyError("data_dict must contain keys 'x' and 'y'")

        self.data_dict = data_dict
        self.X = data_dict['x']
        self.Y = data_dict['y']
        
        # Calculate bounds once
        self.X_min, self.X_max = self.X.min(), self.X.max()
        self.Y_min, self.Y_max = self.Y.min(), self.Y.max()
        self.length = self.X_max - self.X_min
        self.width = self.Y_max - self.Y_min

    def _plot_format(self, plot_func, key_cmap_dict: Union[Dict, List, str], 
                              orientation: str = 'vertical', default_cmap: str = 'viridis') -> plt.Figure:
        
        processed_dict, num_plots = self._keycmap_dict_process(key_cmap_dict, default_cmap=default_cmap)

        if orientation == 'vertical': rows, cols = num_plots, 1
        else: rows, cols = 1, num_plots

        fig, axes = plt.subplots(rows, cols, constrained_layout=True)

        # Ensure axes is iterable even if it's a single plot
        if num_plots == 1:
            axes = [axes]

        for ax, (key, cmap) in zip(axes, processed_dict.items()):
            if key not in self.data_dict:
                print(f"Warning: Key '{key}' not found in data_dict. Skipping.")
                continue

            plot_func(ax=ax, key=key, cmap=cmap)
        
        return fig


    def plot_color(self, datakey_cmap_dict: Union[Dict, List, str], 
                              s: int = 10, orientation: str = 'vertical') -> plt.Figure:
        
        def func(**kwargs):
            ax:plt.Axes = kwargs['ax']
            key = kwargs['key']
            cmap = kwargs['cmap']

            im = ax.scatter(self.X, self.Y, self.data_dict[key], s=s, c=self.data_dict[key], cmap=cmap, marker='s')
            
            ax.set_title(key, fontweight='medium', pad=10, fontsize=13)    
            ax.set_xlabel('x', fontstyle='italic', labelpad=0)
            ax.set_ylabel('y', fontstyle='italic', labelpad=0)
            ax.figure.colorbar(im, ax=ax, pad=0.03, shrink=1.2)

            ax.set_aspect('equal', adjustable='box')

        fig = self._plot_format(func, datakey_cmap_dict, orientation)
        return fig

    def plot(self, datakey_cmap_dict: Union[Dict, List, str], axis: str = 'xy') -> plt.Figure:
        
        def func(**kwargs):
            ax:plt.Axes = kwargs['ax']
            key = kwargs['key']
            cmap = kwargs['cmap']

            if axis in ['x', 'y']:
                ax.plot(self.data_dict[axis], self.data_dict[key], linewidth=2.0, color=cmap)
                ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
                ax.set_xlabel(axis, fontsize=10)
                ax.set_ylabel(key, fontsize=10)
            
            elif axis == 'xy':
                scatter = ax.plot_surface(self.X, self.Y, self.data_dict[key], cmap=cmap, cmap=cmap, marker='.')
                ax.set(xlabel='x', ylabel='y')
                ax.set_title(f'Surface Plot of {key}')
                ax.figure.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

            else:
                raise ValueError("axis must be 'x', 'y', or 'xy'")

        fig = self._plot_format(func, datakey_cmap_dict, default_cmap='viridis')
        return fig

    def plot_distribution(self, key: str, bins: Union[str, int] = 'fd') -> plt.Figure:
        fig, ax = plt.subplots()
        ax.hist(self.data_dict[key], bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(f"{key} distribution")
        return fig

    def plot_loss_curve(self, log_scale: bool = False, linewidth: float = 0.5, 
                        start: int = 0, end: Optional[int] = None, keys: list[str] = ['total_loss', 'bc_loss', 'pde_loss']) -> plt.Figure:
        
        fig, ax = plt.subplots()

        for key in keys:
            ax.plot(self.data_dict[key][start:end], label=key, linewidth=linewidth)

        if log_scale:
            ax.set_yscale("log")
            
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per Iteration")
        ax.legend()  
        
        return fig

    @staticmethod
    def _keycmap_dict_process(key_input: Union[Dict, List, Tuple, str], 
                              default_cmap: str = 'viridis') -> Tuple[Dict, int]:
        """
        Normalizes input into a dictionary of {key: colormap}.
        """
        if isinstance(key_input, dict):
            # Fill None values with default
            return {k: (v if v is not None else default_cmap) for k, v in key_input.items()}, len(key_input)
        
        if isinstance(key_input, (list, tuple)):
            return {k: default_cmap for k in key_input}, len(key_input)

        if isinstance(key_input, str):
            return {key_input: default_cmap}, 1

        raise TypeError("key_cmap_dict must be a dict, list, tuple, or string.")
    
    def _handle_limit(self, ax, range_x=None, range_y=None):
        # Handle Limits
        if range_x:
            ax.set_xlim(range_x[0], range_x[1])
        else:
            margin_x = 0.2 * self.width if self.length < 0.001 else 0
            ax.set_xlim(self.X_min - margin_x, self.X_max + margin_x)

        if range_y:
            ax.set_ylim(range_y[0], range_y[1])
        else:
            margin_y = 0.2 * self.length if self.width < 0.001 else 0
            ax.set_ylim(self.Y_min - margin_y, self.Y_max + margin_y)