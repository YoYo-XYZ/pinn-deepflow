from typing import Dict, List, Optional, Tuple, Union

try:
    import ultraplot as plt
except ImportError:
    import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

class Visualizer:
    """
    Main visualization class for processing data dictionaries.
    Refactored for simplicity and maintainability.
    """
    refwidth_default = 6
    cmap_default = 'viridis'
    color_default = 'blue'

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.data_dict = data_dict
        # Cache coordinates for convenience, if they exist
        self.X = data_dict.get('x')
        self.Y = data_dict.get('y')

    def _create_subplot(self, ref_width = None, ref_height=None):
        fig, ax = plt.subplot(refwidth = (self.refwidth_default if ref_width is None else ref_width), refheight = ref_height)
        return fig, ax

    def plot_color(self, color_axis:str, x_axis:str = 'x', y_axis:str = 'y', cmap = 'viridis', s: Union[int, float] = 2, orientation: str = 'vertical', return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Creates scatter plots (heatmap style) for the specified keys.
        """
        fig, ax = self._create_subplot()

        # Plot
        scatter = ax.scatter(self.data_dict[x_axis], self.data_dict[y_axis], s=s, c=self.data_dict[color_axis], cmap=cmap, marker='s')
        
        # Styling
        ax.format(title = color_axis, xlabel = x_axis, ylabel = y_axis, aspect = 'equal', grid = False)
        ax.set_xlim(self.data_dict[x_axis].min(), self.data_dict[x_axis].max())
        ax.set_ylim(self.data_dict[y_axis].min(), self.data_dict[y_axis].max())
        fig.colorbar(scatter, ax=ax)

        if return_ax:
            return fig, ax
        return fig
    
    # Modern alias
    plot_scatter = plot_color

    def plot(self, z_axis: str = None, x_axis:str = 'x', y_axis:str = 'y', return_ax: bool = False, color = None) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        General plotting method.
        If axis='xy': 3D surface plot.
        If axis='x' or 'y': 1D line plot against that axis.
        """    
        if z_axis is None:
            fig, ax = self._create_subplot()
            # Line Plot
            ax.plot(self.data_dict[x_axis], self.data_dict[y_axis], color=self.color_default if color is None else color)
            ax.grid(True)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
        else:
            fig, ax = plt.subplot(refwidth = self.refwidth_default, proj = '3d')
            # 3D Scatter Plot
            scatter = ax.scatter(self.data_dict[x_axis], self.data_dict[y_axis], self.data_dict[z_axis], c=self.data_dict[z_axis], cmap=self.cmap_default if color is None else color, s=3)
            ax.set(xlabel=x_axis, ylabel=y_axis)
            ax.set_title(f'3D Scatter Plot of {z_axis}')
            colorbar = fig.colorbar(scatter, ax=ax)

        if return_ax:
            return fig, ax
        return fig

    def plot_distribution(self, key: str, bins: Union[str, int] = 'fd', return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Plots histograms for the specified keys.
        """
        fig, ax = self._create_subplot()
        ax.hist(self.data_dict[key], bins=bins)
        ax.set_title(f"{key} distribution")
        
        if return_ax:
            return fig, ax
        return fig

    def plot_loss_curve(self, log_scale: bool = False, 
                        start: int = 0, end: Optional[int] = None, 
                        keys: List[str] = ['total_loss', 'bc_loss', 'pde_loss'], return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Plots loss per iteration.
        """
        fig, ax = plt.subplots(refwidth=5, refheight=3)

        # Plot
        for key in keys:
            if key in self.data_dict:
                ax.plot(self.data_dict[key][start:end], label=key)

        # Styling
        if log_scale: ax.set_yscale("log")
        ax.format(title = "Loss per Iteration", xlabel = "Iteration", ylabel = "Loss")
        ax.legend()  
        
        if return_ax:
            return fig, ax
        return fig
    
### Need interpolation ####

    def plot_contour(self, color_axis:str, x_axis:str = 'x', y_axis:str = 'y', cmap = 'jet', levels = 50, return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Creates scatter plots (heatmap style) for the specified keys.
        """
        fig, ax = self._create_subplot()
        (C), (X, Y) = self._interpolate(color_axis , x_key = x_axis, y_key = y_axis)

        # Plot
        scatter = ax.contourf(X, Y, C, cmap=cmap, levels = levels)
        
        # Styling
        ax.format(title = color_axis, xlabel = x_axis, ylabel = y_axis, aspect = 'equal', grid = False)
        ax.set_xlim(self.data_dict[x_axis].min(), self.data_dict[x_axis].max())
        ax.set_ylim(self.data_dict[y_axis].min(), self.data_dict[y_axis].max())
        fig.colorbar(scatter, ax=ax)

        if return_ax:
            return fig, ax
        return fig
    
    def plot_streamline(self, u:str, v:str, x_axis:str = 'x', y_axis:str = 'y', cmap = 'viridis', levels = 100, return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Creates scatter plots (heatmap style) for the specified keys.
        """
        fig, ax = self._create_subplot()

        (U, V), (X, Y) = self._interpolate(u, v , x_key = x_axis, y_key = y_axis, points=200)

        # Plot
        stream = ax.streamplot(X, Y, U, V, color = (U**2 + V**2)**0.5, cmap = cmap, levels = levels, broken_streamlines = False)
    
        # Styling
        ax.format(title = f"Streamline of {u} and {v}", xlabel = x_axis, ylabel = y_axis, aspect = 'equal', grid = False)
        ax.set_xlim(self.data_dict[x_axis].min(), self.data_dict[x_axis].max())
        ax.set_ylim(self.data_dict[y_axis].min(), self.data_dict[y_axis].max())
        fig.colorbar(stream.lines, ax=ax)

        if return_ax:
            return fig, ax
        return fig

    def _interpolate(self, *keys, x_key = 'x', y_key='y', points = None):
        """
        Interpolates scattered data onto a grid for surface plotting.
        """
        x = self.data_dict[x_key]
        y = self.data_dict[y_key]
        ratio = (np.max(y) - np.min(y)) / (np.max(x) - np.min(x))
        
        if points is None: points = len(x)
        n_x = int(np.sqrt(points / ratio))
        n_y = int(ratio * n_x)

        xi = np.linspace(np.min(x), np.max(x), n_x)
        yi = np.linspace(np.min(y), np.max(y), n_y)
        X, Y = np.meshgrid(xi, yi)

        return [interpolate.griddata((x, y), self.data_dict[key], (X, Y), method='cubic') for key in keys], (X, Y)