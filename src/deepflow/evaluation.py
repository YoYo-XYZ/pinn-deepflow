"""PINN and FEM evaluation on individual geometries."""

from typing import List, Union, Dict, Any, Optional

import torch
import numpy as np
import ultraplot as plt

from .nn import PINN 
from .geometry import Area, Bound, CustomData
from .visualization import Visualizer


class Evaluator(Visualizer):
    """Evaluate a PINN on one geometry and prepare plotting data.

    Args:
        pinns_model: PINN model used to calculate predictions and residuals.
        geometry: Sampled ``Area``, ``Bound``, or ``CustomData``
            object.

    Attributes:
        data_dict: Materialized NumPy fields, coordinates, residuals, and loss
            history.

    Notes:
        Sampling and time changes replace ``data_dict``. Reassign any custom
        fields after refreshing the evaluator.
    """

    def __init__(self, pinns_model: PINN, geometry: Area|Bound|CustomData) -> None:
        """
        Initialize an evaluator for a model and geometry.

        Args:
            pinns_model: The physics-informed neural network model.
            geometry: The geometric domain or custom data to evaluate.
        """
        self.model = pinns_model 
        self.geometry = geometry

        self.data_dict: Dict[str, Any] = {}
        self.is_postprocessed = False
        
        if isinstance(geometry, CustomData):
            self.postprocess()
        
        if hasattr(self.geometry, "X") and getattr(self.geometry, "X") is not None:
            self.postprocess()

    def sampling_line(self, n_points: int, scheme: str = 'uniform') -> None:
        """Sample a boundary and refresh evaluated fields.

        Args:
            n_points: Number of boundary points.
            scheme: ``"uniform"``, ``"random"``, or ``"lhs"``.
        """
        self.geometry.sampling_line(n_points, scheme)
        self.postprocess()

    def sampling_area(self, res_list: List[int], scheme: str = 'uniform') -> None:
        """Sample an area and refresh evaluated fields.

        Args:
            res_list: Area resolution as a scalar or ``[nx, ny]`` pair.
            scheme: ``"uniform"``, ``"random"``, or ``"lhs"``.
        """
        self.geometry.sampling_area(res_list, scheme)
        self.postprocess()

    def define_time(
        self,
        range_t: Union[float, int, List[float]],
        sampling_scheme: str = "uniform",
        expo_scaling: Optional[bool] = None,
    ) -> None:
        """Define time coordinates and refresh evaluated fields.

        Args:
            range_t: Time interval or fixed time value.
            sampling_scheme: Time sampling scheme.
            expo_scaling: Whether to apply exponential time scaling.
        """
        self.geometry.define_time(
            range_t,
            sampling_scheme=sampling_scheme,
            expo_scaling=expo_scaling,
        )
        self.postprocess()
    def postprocess(self) -> None:
        """
        Recompute predictions, residuals, and coordinates.

        Returns:
            None. ``data_dict`` is replaced with NumPy-compatible values.
        """
        self.geometry.process_coordinates()

        self._create_data_dict()
        self.is_postprocessed = True
        
        super().__init__(self.data_dict)

    def _create_data_dict(self) -> Dict[str, Any]:
        self.model.eval()

        data_dict = self.geometry.process_model(self.model)

        if self.geometry.physics_type == 'PDE':
            data_dict[f"{self.geometry.physics_type}".lower() + "_residual"] = self.geometry.calc_residual_field(self.model)
            data_dict.update(self.geometry.PDE.var)
        
        elif self.geometry.physics_type in ['BC', 'IC']:
            data_dict[f"{self.geometry.physics_type}".lower() + "_residual"] = self.geometry.calc_residual_field(self.model)
        data_dict['x'] = self.geometry.X
        data_dict['y'] = self.geometry.Y
        if self.geometry.range_t:
            data_dict['t'] = self.geometry.T

        data_dict.update(self.model.loss_history)

        self.data_dict = self._convert_to_numpy(data_dict)

        return self.data_dict

    def _convert_to_numpy(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        clean_dict = {}
        for key, value in data.items():
            try:
                if isinstance(value, list):
                    clean_dict[key] = np.array(value)
                elif isinstance(value, torch.Tensor):
                    clean_dict[key] = value.detach().cpu().numpy()
                elif isinstance(value, np.ndarray):
                    clean_dict[key] = value
                else:
                    clean_dict[key] = np.array(value)
            except Exception as e:
                print(f"Warning: Could not convert key '{key}' to numpy. Error: {e}")
                clean_dict[key] = value
        return clean_dict

    def __getitem__(self, key: str) -> Any:
        return self.data_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.data_dict[key] = value

    def __str__(self):
        return f"Available data keys: {tuple(self.data_dict.keys())}"
    
    def plot_animate(self, color_axis:str, x_axis: str = 'x', y_axis: str = 'y', cmap = 'viridis', range_t=None, dt=None, frame_interval = 10, plot_type: str = 'scatter', s = 6, color_range:list=None) -> Any:
        """
        Create and display an animation over time for a field.

        Args:
            color_axis: Field to animate.
            x_axis: x-coordinate field.
            y_axis: y-coordinate field.
            cmap: Matplotlib colormap name.
            range_t: Two-element animation time interval.
            dt: Time increment between frames.
            frame_interval: Delay between frames in milliseconds.
            plot_type: ``"scatter"``, ``"tripcolor"``, or ``"contourf"``.
            s: Scatter marker size.
            color_range: Optional ``[min, max]`` color limits.

        Returns:
            The created Matplotlib animation object.
        """
        import matplotlib.animation as animation

        fig, ax = self._create_equal_aspect_subplot(
            self.data_dict.get(x_axis), self.data_dict.get(y_axis)
        )
        ax.format(grid=False)

        color_list = []
        time_list = list(np.arange(range_t[0], range_t[1], dt))
        for t in time_list:
            self.define_time(t)
            color_list.append(self.data_dict[color_axis])
        if color_range:
            min_val, max_val = color_range
        else:
            max_val = np.max([np.max(c) for c in color_list])
            min_val = np.min([np.min(c) for c in color_list])

        if plot_type == 'scatter':
            plot = ax.scatter(self.data_dict[x_axis], self.data_dict[y_axis], c=color_list[0], cmap=cmap, vmin=min_val, vmax=max_val, marker='s', s = s)
        elif plot_type == 'tripcolor':
            plot = ax.tripcolor(self.data_dict[x_axis], self.data_dict[y_axis], color_list[0], cmap=cmap, vmin=min_val, vmax=max_val, shading = 'gouraud')
        elif plot_type == 'contourf':
            plot = ax.tricontourf(self.data_dict[x_axis], self.data_dict[y_axis], color_list[0], cmap=cmap, vmin=min_val, vmax=max_val, levels=100)

        title = ax.set_title(f'{color_axis} - Time: {time_list[0]:.3f}')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_aspect('equal')
        
        ax.colorbar(plot, ax=ax)
        
        def animate(frame):
            plot.set_array(color_list[frame].ravel())
            title.set_text(f'{color_axis} - Time: {time_list[frame]:.3f}')
            return plot, title

        ani = animation.FuncAnimation(fig, animate, frames=len(time_list), interval=frame_interval, blit=True)
        plt.show()
        return ani


class ReferenceEvaluator(Evaluator):
    """Evaluate a solved ``ReferenceSolution`` on one geometry.

    Reference evaluators deliberately do not calculate PINN residuals or copy
    model training history.  They retain the FEM solution and re-query it
    whenever the geometry is sampled again or its time coordinates change.

    Args:
        reference_solution: Solved ``ReferenceSolution`` queried by this
            evaluator.
        geometry: Geometry whose coordinates are used for FEM queries.
    """

    def __init__(self, reference_solution, geometry: Area | Bound | CustomData) -> None:
        """Initialize an evaluator backed by a reference solution."""
        self.model = None
        self.reference_solution = reference_solution
        self.geometry = geometry
        self.data_dict: Dict[str, Any] = {}
        self.is_postprocessed = False
        self.metadata = reference_solution.metadata

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _same_coordinates(current, processed) -> bool:
        if current is None or processed is None:
            return False
        current_array = ReferenceEvaluator._as_numpy(current)
        processed_array = ReferenceEvaluator._as_numpy(processed)
        return (
            current_array.shape == processed_array.shape
            and np.array_equal(current_array, processed_array)
        )

    def _prepare_coordinates(self) -> None:
        if getattr(self.geometry, "X", None) is None or getattr(
            self.geometry, "Y", None
        ) is None:
            raise ValueError(
                "Cannot evaluate an unsampled geometry: coordinates X and Y "
                "must be defined first."
            )

        x_processed = getattr(self.geometry, "X_", None)
        y_processed = getattr(self.geometry, "Y_", None)
        if not self._same_coordinates(self.geometry.X, x_processed) or not self._same_coordinates(
            self.geometry.Y, y_processed
        ):
            self._process_coordinates_preserving_time()
            return

        if getattr(self.reference_solution, "is_transient", False) and not getattr(
            self.reference_solution, "time_from_y", False
        ):
            t_values = self._time_values()
            if t_values is None:
                if getattr(self.geometry, "range_t", None) is not None:
                    self._process_coordinates_preserving_time()
            else:
                try:
                    np.broadcast(
                        np.empty(np.shape(t_values)),
                        np.empty(np.shape(self.geometry.X)),
                    )
                except ValueError:
                    if getattr(self.geometry, "range_t", None) is not None:
                        self._process_coordinates_preserving_time()

    def _time_values(self):
        for name in ("T", "t", "T_"):
            value = getattr(self.geometry, name, None)
            if value is not None:
                return value
        return None

    def _process_coordinates_preserving_time(self) -> None:
        existing_time = self._time_values()
        existing_shape = (
            None
            if existing_time is None
            else tuple(self._as_numpy(existing_time).shape)
        )

        self.geometry.process_coordinates()

        if (
            existing_time is None
            or existing_shape is None
            or existing_shape != tuple(self._as_numpy(self.geometry.X).shape)
        ):
            return

        self.geometry.t = existing_time
        self.geometry.T = existing_time
        if isinstance(existing_time, torch.Tensor):
            restored = existing_time.detach().clone()
            if isinstance(getattr(self.geometry, "X_", None), torch.Tensor):
                restored = restored.to(self.geometry.X_.device)
            self.geometry.T_ = restored.requires_grad_()
            if isinstance(
                getattr(self.geometry, "inputs_tensor_dict", None), dict
            ):
                self.geometry.inputs_tensor_dict["t"] = self.geometry.T_
        else:
            self.geometry.T_ = existing_time

    def define_time(
        self,
        range_t: Union[float, int, List[float]],
        sampling_scheme: str = "uniform",
        expo_scaling: Optional[bool] = None,
    ) -> None:
        """Update time coordinates and re-query the FEM fields.

        Args:
            range_t: Time interval or fixed time value.
            sampling_scheme: Time sampling scheme.
            expo_scaling: Whether to apply exponential time scaling.
        """
        self.geometry.define_time(
            range_t,
            sampling_scheme=sampling_scheme,
            expo_scaling=expo_scaling,
        )
        self.geometry.process_coordinates()
        self.postprocess()

    def _create_data_dict(self) -> Dict[str, Any]:
        self._prepare_coordinates()

        x = self._as_numpy(self.geometry.X)
        y = self._as_numpy(self.geometry.Y)
        solution = self.reference_solution
        fields = getattr(solution, "fields", None)
        if fields is None:
            fields = solution.declared_fields
        fields = tuple(fields)

        if getattr(solution, "is_transient", False) and not getattr(
            solution, "time_from_y", False
        ):
            time_values = self._time_values()
            if time_values is None:
                raise ValueError(
                    "Transient FEM evaluation requires existing time "
                    "coordinates on every geometry."
                )
            try:
                np.broadcast(
                    np.empty(np.shape(time_values)),
                    np.empty(np.shape(self.geometry.X)),
                )
            except ValueError as exc:
                raise ValueError(
                    "Transient FEM time coordinates must align with the "
                    "geometry's spatial coordinates."
                ) from exc
            t = self._as_numpy(time_values)
            values = solution.evaluate(x, y, t=t, fields=fields)
        else:
            t = None
            values = solution.evaluate(x, y, fields=fields)

        data_dict = {
            f"{name}_ref": np.asarray(value)
            for name, value in values.items()
        }
        data_dict["x"] = x
        data_dict["y"] = y
        if t is not None:
            data_dict["t"] = t

        self.data_dict = self._convert_to_numpy(data_dict)
        return self.data_dict

    def postprocess(self) -> None:
        """Re-query the reference fields at the current geometry coordinates."""
        self._create_data_dict()
        self.is_postprocessed = True
        Visualizer.__init__(self, self.data_dict)


class ReferenceGroupEvaluator:
    """FEM results organized into geometry-specific reference evaluators.

    Access fields and plots through ``area_list`` and ``bound_list``.
    The solved FEM solution is available as ``reference_solution`` for
    arbitrary point queries and export.
    """

    def __init__(self, reference_solution, domain) -> None:
        self.reference_solution = reference_solution
        self.metadata = reference_solution.metadata
        self.area_list = [
            ReferenceEvaluator(reference_solution, geometry)
            for geometry in dict.fromkeys(domain.area_list)
        ]
        self.bound_list = [
            ReferenceEvaluator(reference_solution, geometry)
            for geometry in dict.fromkeys(domain.bound_list)
        ]
