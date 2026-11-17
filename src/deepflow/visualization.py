"""Plotting helpers for evaluated DeepFlow fields."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import ultraplot as plt
from scipy import interpolate
from scipy.spatial import QhullError


class Visualizer:
    """Plot named fields and coordinates from NumPy arrays in ``data_dict``.

    Axis and field arguments select dictionary keys. Plot methods return a
    figure, or ``(figure, axes)`` when ``return_ax=True``.
    """
    refwidth_default = 6
    refheight_max_default = None
    cmap_default = 'viridis'
    color_default = 'blue'

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        """Initialize a visualizer from evaluated data."""
        self.data_dict = data_dict
        self.X = data_dict.get('x')
        self.Y = data_dict.get('y')

    def _create_subplot(self, ref_width = None, ref_height=None):
        fig, ax = plt.subplot(refwidth = (self.refwidth_default if ref_width is None else ref_width), refheight = ref_height)
        return fig, ax

    def _create_equal_aspect_subplot(self, x_values=None, y_values=None, ref_width=None, ref_height=None):
        max_width = self.refwidth_default if ref_width is None else ref_width
        if ref_height is not None:
            max_height = ref_height
        elif self.refheight_max_default is not None:
            max_height = self.refheight_max_default
        else:
            max_height = max_width
        try:
            x = np.asarray(x_values).reshape(-1) if x_values is not None else None
            y = np.asarray(y_values).reshape(-1) if y_values is not None else None
            if x is None or y is None or x.size == 0 or y.size == 0:
                raise ValueError("fallback")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("fallback")
            x_range = np.ptp(x)
            y_range = np.ptp(y)
            if not np.isfinite(x_range) or not np.isfinite(y_range):
                raise ValueError("fallback")
            if x_range <= 0 or y_range <= 0 or max_width <= 0 or max_height <= 0:
                raise ValueError("fallback")
            refwidth_eff = min(max_width, max_height / (y_range / x_range))
        except Exception:
            return self._create_subplot(ref_width, ref_height)
        subplot_kwargs = {"ref_width": refwidth_eff}
        if ref_height is not None or self.refheight_max_default is not None:
            subplot_kwargs["ref_height"] = max_height
        return self._create_subplot(**subplot_kwargs)

    def plot_color(self, color_axis: str, x_axis: str = 'x', y_axis: str = 'y', cmap='viridis', s: Union[int, float] = 2, return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """Scatter ``x_axis`` and ``y_axis``, colored by ``color_axis``.

        ``cmap`` selects the colormap; ``s`` sets marker size.
        """
        fig, ax = self._create_equal_aspect_subplot(
            self.data_dict.get(x_axis), self.data_dict.get(y_axis)
        )

        scatter = ax.scatter(self.data_dict[x_axis], self.data_dict[y_axis], s=s, c=self.data_dict[color_axis], cmap=cmap, marker='s')
        
        ax.format(title = color_axis, xlabel = x_axis, ylabel = y_axis, aspect = 'equal', grid = False)
        ax.set_xlim(self.data_dict[x_axis].min(), self.data_dict[x_axis].max())
        ax.set_ylim(self.data_dict[y_axis].min(), self.data_dict[y_axis].max())
        fig.colorbar(scatter, ax=ax)

        if return_ax:
            return fig, ax
        return fig
    
    plot_scatter = plot_color

    def plot(self, z_axis: str = None, x_axis:str = 'x', y_axis:str = 'y', return_ax: bool = False, color = None) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """Plot ``y_axis`` against ``x_axis``, or scatter in 3D when ``z_axis`` is
        set.

        ``color`` is a line color for 2D or a colormap name for 3D.
        """
        if z_axis is None:
            fig, ax = self._create_subplot()
            ax.plot(self.data_dict[x_axis], self.data_dict[y_axis], color=self.color_default if color is None else color)
            ax.grid(True)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
        else:
            fig, ax = plt.subplot(refwidth = self.refwidth_default, proj = '3d')
            scatter = ax.scatter(self.data_dict[x_axis], self.data_dict[y_axis], self.data_dict[z_axis], c=self.data_dict[z_axis], cmap=self.cmap_default if color is None else color, s=3)
            ax.set(xlabel=x_axis, ylabel=y_axis)
            ax.set_title(f'3D Scatter Plot of {z_axis}')
            fig.colorbar(scatter, ax=ax)

        if return_ax:
            return fig, ax
        return fig

    def plot_distribution(self, key: str, bins: Union[str, int] = 'fd', return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """Plot a histogram of ``key``; ``bins`` is a count or NumPy binning
        strategy.
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
        """Plot available loss-history ``keys`` over iterations.

        ``start:end`` selects history indices, excluding ``end``. ``log_scale``
        uses a logarithmic y-axis.
        """
        fig, ax = plt.subplots(refwidth=5, refheight=3)

        for key in keys:
            if key in self.data_dict:
                values = np.asarray(self.data_dict[key]).reshape(-1)
                iterations = np.arange(values.size)[start:end]
                ax.plot(iterations, values[start:end], label=key)

        if log_scale:
            ax.set_yscale("log")
        ax.format(title = "Loss per Iteration", xlabel = "Iteration", ylabel = "Loss")
        ax.legend()  
        
        if return_ax:
            return fig, ax
        return fig
    
    def plot_contour(self, color_axis:str, x_axis:str = 'x', y_axis:str = 'y', cmap = 'jet', levels = 50, return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """Plot interpolated ``color_axis`` contours on ``x_axis`` and ``y_axis``.

        ``cmap`` selects the colormap; ``levels`` sets the contour count.
        """
        fig, ax = self._create_equal_aspect_subplot(
            self.data_dict.get(x_axis), self.data_dict.get(y_axis)
        )
        (C,), (X, Y) = self._interpolate(color_axis, x_key=x_axis, y_key=y_axis)

        scatter = ax.contourf(X, Y, C, cmap=cmap, levels = levels)
        
        ax.format(title = color_axis, xlabel = x_axis, ylabel = y_axis, aspect = 'equal', grid = False)
        ax.set_xlim(self.data_dict[x_axis].min(), self.data_dict[x_axis].max())
        ax.set_ylim(self.data_dict[y_axis].min(), self.data_dict[y_axis].max())
        fig.colorbar(scatter, ax=ax)

        if return_ax:
            return fig, ax
        return fig
    
    def plot_streamline(self, u:str, v:str, x_axis:str = 'x', y_axis:str = 'y', cmap = 'viridis', levels = 100, return_ax: bool = False) -> Union[plt.Figure, Tuple[plt.Figure, object]]:
        """Plot interpolated vector fields ``u`` and ``v`` on ``x_axis`` and
        ``y_axis``.

        ``cmap`` and ``levels`` are passed to the streamline plot.
        """
        fig, ax = self._create_equal_aspect_subplot(
            self.data_dict.get(x_axis), self.data_dict.get(y_axis)
        )

        (U, V), (X, Y) = self._interpolate(u, v, x_key=x_axis, y_key=y_axis, points=2000)

        stream = ax.streamplot(X, Y, U, V, color = (U**2 + V**2)**0.5, cmap = cmap, levels = levels, broken_streamlines = False)
    
        ax.format(title = f"Streamline of {u} and {v}", xlabel = x_axis, ylabel = y_axis, aspect = 'equal', grid = False)
        ax.set_xlim(self.data_dict[x_axis].min(), self.data_dict[x_axis].max())
        ax.set_ylim(self.data_dict[y_axis].min(), self.data_dict[y_axis].max())
        fig.colorbar(stream.lines, ax=ax)

        if return_ax:
            return fig, ax
        return fig

    def _interpolate(self, *keys, x_key='x', y_key='y', points=None):
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
