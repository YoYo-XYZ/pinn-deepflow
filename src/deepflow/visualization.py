"""Plotting helpers for evaluated DeepFlow fields."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import ultraplot as plt
from scipy import interpolate
from scipy.spatial import QhullError


class Visualizer:
    """Create plots from a mapping of field names to NumPy arrays.

    Args:
        data_dict: Evaluated fields and coordinates. Plot methods expect the
            requested field keys and usually ``x`` and ``y`` coordinates.
    """
    refwidth_default = 6
    cmap_default = 'viridis'
    color_default = 'blue'

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        """Initialize a visualizer from evaluated data."""
        self.data_dict = data_dict
        # Cache coordinates for convenience, if they exist
        self.X = data_dict.get('x')
        self.Y = data_dict.get('y')

    def _create_subplot(self, ref_width = None, ref_height=None):
        fig, ax = plt.subplot(refwidth = (self.refwidth_default if ref_width is None else ref_width), refheight = ref_height)
        return fig, ax

    def plot_color(self, color_axis: str, x_axis: str = 'x', y_axis: str = 'y', cmap='viridis', s: Union[int, float] = 2, return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Create a scatter plot colored by a field.

        Args:
            color_axis: Field used for point colors.
            x_axis: Field used for horizontal coordinates.
            y_axis: Field used for vertical coordinates.
            cmap: Matplotlib colormap name.
            s: Marker size.
            return_ax: Return ``(figure, axes)`` instead of only the figure.

        Returns:
            A figure, or a ``(figure, axes)`` tuple when ``return_ax`` is true.
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
        Create a line plot or a three-dimensional scatter plot.

        Args:
            z_axis: Optional field for a 3-D scatter plot. If omitted, plot
                ``y_axis`` against ``x_axis`` as a line.
            x_axis: Field used for horizontal coordinates.
            y_axis: Field used for vertical coordinates.
            return_ax: Return ``(figure, axes)`` instead of only the figure.
            color: Optional line color or colormap name.

        Returns:
            A figure, or a ``(figure, axes)`` tuple when ``return_ax`` is true.
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
            fig.colorbar(scatter, ax=ax)

        if return_ax:
            return fig, ax
        return fig

    def plot_distribution(self, key: str, bins: Union[str, int] = 'fd', return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Plot a histogram for a field.

        Args:
            key: Field to plot.
            bins: Number of bins or a NumPy binning strategy.
            return_ax: Return ``(figure, axes)`` instead of only the figure.

        Returns:
            A figure, or a ``(figure, axes)`` tuple when ``return_ax`` is true.
        """
        fig, ax = self._create_subplot()
        ax.hist(self.data_dict[key], bins=bins)
        ax.set_title(f"{key} distribution")
        
        if return_ax:
            return fig, ax
        return fig

    def plot_loss_curve(self, log_scale: bool = True, 
                        start: int = 0, end: Optional[int] = None, 
                        keys: Tuple[str, ...] = ('total_loss', 'bc_loss', 'pde_loss'), return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Plot recorded losses over training iterations.

        Args:
            log_scale: Use a logarithmic y-axis.
            start: First history index to display.
            end: Exclusive final history index.
            keys: Loss-history keys to plot when present.
            return_ax: Return ``(figure, axes)`` instead of only the figure.

        Returns:
            A figure, or a ``(figure, axes)`` tuple when ``return_ax`` is true.
        """
        fig, ax = plt.subplots(refwidth=5, refheight=3)

        # Plot
        for key in keys:
            if key in self.data_dict:
                values = np.asarray(self.data_dict[key]).reshape(-1)
                iterations = np.arange(values.size)[start:end]
                ax.plot(iterations, values[start:end], label=key)

        # Styling
        if log_scale:
            ax.set_yscale("log")
        ax.format(title = "Loss per Iteration", xlabel = "Iteration", ylabel = "Loss")
        ax.legend()  
        
        if return_ax:
            return fig, ax
        return fig
    
    def plot_contour(self, color_axis:str, x_axis:str = 'x', y_axis:str = 'y', cmap = 'jet', levels = 50, return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """
        Create a filled contour plot for an interpolated field.

        Args:
            color_axis: Field to interpolate and plot.
            x_axis: x-coordinate field.
            y_axis: y-coordinate field.
            cmap: Matplotlib colormap name.
            levels: Number of contour levels.
            return_ax: Return ``(figure, axes)`` instead of only the figure.

        Returns:
            A figure, or a ``(figure, axes)`` tuple when ``return_ax`` is true.
        """
        fig, ax = self._create_subplot()
        (C,), (X, Y) = self._interpolate(color_axis, x_key=x_axis, y_key=y_axis)

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
        Create a streamline plot for two interpolated vector components.

        Args:
            u: Field containing the x component.
            v: Field containing the y component.
            x_axis: x-coordinate field.
            y_axis: y-coordinate field.
            cmap: Matplotlib colormap name.
            levels: Streamline density or plotting level configuration.
            return_ax: Return ``(figure, axes)`` instead of only the figure.

        Returns:
            A figure, or a ``(figure, axes)`` tuple when ``return_ax`` is true.
        """
        fig, ax = self._create_subplot()

        (U, V), (X, Y) = self._interpolate(u, v, x_key=x_axis, y_key=y_axis, points=2000)

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

    def _interpolate(self, *keys, x_key='x', y_key='y', points=None):
        """
        Interpolate scattered fields onto a rectangular grid.
        """
        x = np.asarray(self.data_dict[x_key]).reshape(-1)
        y = np.asarray(self.data_dict[y_key]).reshape(-1)

        if x.size == 0 or y.size == 0:
            raise ValueError("Interpolation requires non-empty x and y coordinates.")
        if x.size != y.size:
            raise ValueError("Interpolation requires x and y coordinates with equal lengths.")
        if x.size < 3:
            raise ValueError("Interpolation requires at least three 2-D samples.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Interpolation requires finite x and y coordinates.")

        x_range = np.ptp(x)
        y_range = np.ptp(y)
        if x_range <= 0 or y_range <= 0:
            raise ValueError("Interpolation requires varying x and y coordinates.")

        points = x.size if points is None else int(points)
        if points < 4:
            raise ValueError("Interpolation requires enough points for a 2x2 grid.")

        fields = []
        for key in keys:
            field = np.asarray(self.data_dict[key]).reshape(-1)
            if field.size != x.size:
                raise ValueError(
                    f"Interpolation field {key!r} must match the coordinate length."
                )
            fields.append(field)

        ratio = y_range / x_range
        n_x = max(2, int(np.sqrt(points / ratio)))
        n_y = max(2, int(ratio * n_x))

        xi = np.linspace(x.min(), x.max(), n_x)
        yi = np.linspace(y.min(), y.max(), n_y)
        X, Y = np.meshgrid(xi, yi)

        try:
            values = [
                interpolate.griddata((x, y), field, (X, Y), method='cubic')
                for field in fields
            ]
        except QhullError as exc:
            raise ValueError(
                "Cubic interpolation requires non-collinear 2-D samples."
            ) from exc

        return values, (X, Y)
