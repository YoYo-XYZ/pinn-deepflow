from typing import List, Union, Dict, Any, Optional

import torch
import numpy as np
try:
    import ultraplot as plt
except ImportError:
    import matplotlib.pyplot as plt

# Explicit imports are better for open source
from .nn import PINN 
from .geometry import Area, Bound, CustomData
from .visualization import Visualizer

class Evaluator(Visualizer):
    """
    Evaluates a PINN model against a given geometry and prepares data for visualization.
    """

    def __init__(self, pinns_model: PINN, geometry: Area|Bound|CustomData) -> None:
        """
        Args:
            pinns_model: The physics-informed neural network model.
            geometry: The geometric domain (Area or Bound) to evaluate on.
            custom_data: Optional custom data to include in the evaluation.
        """
        self.model = pinns_model 
        self.geometry = geometry

        # Initialize internal state
        self.data_dict: Dict[str, Any] = {}
        self.is_postprocessed = False
        
        if isinstance(geometry, CustomData):
            self.postprocess()
        
        # If geometry already has coordinates (i.e. has been sampled), we can attempt postprocess
        if hasattr(self.geometry, "X") and getattr(self.geometry, "X") is not None:
            self.postprocess()
        # If Visualizer requires data_dict immediately, initialize it with empty data
        # or handle the parent init carefully.
        # super().__init__({}) 

    def sampling_line(self, n_points: int, scheme: str = 'uniform') -> None:
        """Samples points along a line within the geometry."""
        self.geometry.sampling_line(n_points, scheme)
        self.postprocess()

    def sampling_area(self, res_list: List[int], scheme: str = 'uniform') -> None:
        """Samples points within the area of the geometry."""
        self.geometry.sampling_area(res_list, scheme)
        self.postprocess()

    def define_time(
        self,
        range_t: Union[float, int, List[float]],
        sampling_scheme: str = "uniform",
        expo_scaling: Optional[bool] = None,
    ) -> None:
        """Defines time coordinates for transient problems."""
        self.geometry.define_time(
            range_t,
            sampling_scheme=sampling_scheme,
            expo_scaling=expo_scaling,
        )
        self.postprocess()
    def postprocess(self) -> None:
        """
        Aggregates model predictions, residuals, and coordinates, 
        then converts them to NumPy for visualization.
        """
        self.geometry.process_coordinates()

        self._create_data_dict()
        self.is_postprocessed = True
        
        # Initialize the parent Visualizer with the processed data
        super().__init__(self.data_dict)

    def _create_data_dict(self) -> Dict[str, Any]:
        """Internal method to build the dictionary of fields."""
        # Ensure model is in eval mode to disable dropout/batchnorm updates
        self.model.eval()

        # 1. Base model outputs
        data_dict = self.geometry.process_model(self.model)

        # 2. Physics Residuals
        if self.geometry.physics_type == 'PDE':
            # Use .update() for dictionary merging (compatible with older python)
            data_dict[f"{self.geometry.physics_type}".lower() + "_residual"] = self.geometry.calc_residual_field(self.model)
            data_dict.update(self.geometry.PDE.var)
        
        elif self.geometry.physics_type in ['BC', 'IC']:
            data_dict[f"{self.geometry.physics_type}".lower() + "_residual"] = self.geometry.calc_residual_field(self.model)
        # 4. Coordinates
        data_dict['x'] = self.geometry.X
        data_dict['y'] = self.geometry.Y
        if self.geometry.range_t:
            data_dict['t'] = self.geometry.T

        # 5. Training History
        data_dict.update(self.model.loss_history)

        # 6. Normalize to NumPy
        self.data_dict = self._convert_to_numpy(data_dict)

        return self.data_dict

    def _convert_to_numpy(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Helper to safely convert dictionary values to NumPy arrays."""
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
                    # Fallback for scalars or other types
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
        Creates an animation over time for the specified key(s).
        """
        import matplotlib.animation as animation

        fig, ax = plt.subplot(refwidth = Visualizer.refwidth_default, grid=False)

        # Prepare data to animate
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

        # Initialize figure
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
        
        # Add colorbar
        ax.colorbar(plot, ax=ax)
        
        def animate(frame):
            plot.set_array(color_list[frame].ravel())
            title.set_text(f'{color_axis} - Time: {time_list[frame]:.3f}')
            return plot, title

        ani = animation.FuncAnimation(fig, animate, frames=len(time_list), interval=frame_interval, blit=True)
        plt.show()
        return ani


class GroupEvaluator:
    """
    Coordinates geometry-specific :class:`Evaluator` instances for a domain.

    Results remain structured per geometry.  The group does not merge data
    dictionaries because different geometries can represent different physics
    types and therefore expose different residual fields.
    """

    def __init__(self, pinns_model: PINN, domain) -> None:
        self.model = pinns_model
        self.domain = domain

        self._area_entries = self._unique_entries(
            getattr(domain, "area_list", [])
        )
        self._bound_entries = self._unique_entries(
            getattr(domain, "bound_list", [])
        )

        self.area_geometries = [geometry for _, geometry in self._area_entries]
        self.bound_geometries = [geometry for _, geometry in self._bound_entries]

        self._validate_sampled()

        self.area_evaluators = [
            Evaluator(self.model, geometry)
            for _, geometry in self._area_entries
        ]
        self.bound_evaluators = [
            Evaluator(self.model, geometry)
            for _, geometry in self._bound_entries
        ]

        self._evaluators_by_id = {
            id(geometry): evaluator
            for geometry, evaluator in zip(
                self.bound_geometries + self.area_geometries,
                self.bound_evaluators + self.area_evaluators,
            )
        }
        self._ordered_evaluators = [
            *self.bound_evaluators,
            *self.area_evaluators,
        ]

    @staticmethod
    def _unique_entries(geometries):
        """Return first-occurrence ``(index, geometry)`` pairs by identity."""
        seen = set()
        entries = []
        for index, geometry in enumerate(geometries):
            geometry_id = id(geometry)
            if geometry_id in seen:
                continue
            seen.add(geometry_id)
            entries.append((index, geometry))
        return entries

    def _validate_sampled(self) -> None:
        """Raise a clear error when any included geometry lacks coordinates."""
        missing = []
        for category, entries in (
            ("bound", self._bound_entries),
            ("area", self._area_entries),
        ):
            for index, geometry in entries:
                if (
                    getattr(geometry, "X", None) is None
                    or getattr(geometry, "Y", None) is None
                ):
                    missing.append(
                        f"{category}[{index}] ({type(geometry).__name__})"
                    )

        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "Cannot evaluate unsampled domain geometries: "
                f"{joined}. Sample every geometry before calling evaluate()."
            )

    def get_evaluator(self, geometry) -> Evaluator:
        """Return the child evaluator for an original geometry object."""
        evaluator = self._evaluators_by_id.get(id(geometry))
        if evaluator is None or evaluator.geometry is not geometry:
            raise KeyError("Geometry is not part of this GroupEvaluator.")
        return evaluator

    def __iter__(self):
        """Iterate in the same bound-first order as ``ProblemDomain``."""
        return iter(self._ordered_evaluators)

    def __len__(self) -> int:
        return len(self._ordered_evaluators)

    def __str__(self) -> str:
        return (
            "GroupEvaluator("
            f"bound_evaluators={len(self.bound_evaluators)}, "
            f"area_evaluators={len(self.area_evaluators)})"
        )

    def postprocess(self) -> None:
        """Refresh all child evaluators after sampling or model changes."""
        self._validate_sampled()
        for evaluator in self._ordered_evaluators:
            evaluator.postprocess()

    def refresh(self) -> None:
        """Alias for :meth:`postprocess`."""
        self.postprocess()

    @staticmethod
    def _normalize_resolutions(
        resolution,
        count: int,
        name: str,
        *,
        allow_area_pair: bool = False,
    ) -> list:
        """
        Normalize scalar or aligned resolution values for child geometries.

        With one Area, ``[nx, ny]`` remains a valid area resolution.  With
        multiple Areas, nested values are required for per-area ``[nx, ny]``
        resolutions, while a flat list is interpreted as one scalar per Area.
        """
        if count == 0:
            return []

        if not isinstance(resolution, (list, tuple)):
            return [resolution] * count

        values = list(resolution)
        if count == 1:
            if len(values) == 1:
                return [values[0]]
            if allow_area_pair and len(values) == 2:
                return [values]
            raise ValueError(
                f"{name} must be a scalar, a single resolution, or a "
                "two-value [nx, ny] resolution for one Area."
            )

        if len(values) != count:
            raise ValueError(
                f"{name} must contain one resolution per child geometry "
                f"(expected {count}, got {len(values)})."
            )

        if not allow_area_pair and any(
            isinstance(value, (list, tuple)) for value in values
        ):
            raise ValueError(
                f"{name} must contain scalar point counts for Bound children."
            )

        return values

    def sampling_line(self, n_points, scheme: str = "uniform") -> None:
        """Sample and refresh every Bound child."""
        resolutions = self._normalize_resolutions(
            n_points,
            len(self.bound_evaluators),
            "n_points",
        )
        for evaluator, resolution in zip(self.bound_evaluators, resolutions):
            evaluator.sampling_line(resolution, scheme=scheme)

    def sampling_area(self, res_list, scheme: str = "uniform") -> None:
        """
        Sample and refresh every Area child.

        ``CustomData`` entries are intentionally left unchanged because they
        represent fixed user-provided coordinates rather than a sampler.
        """
        area_pairs = [
            (geometry, evaluator)
            for geometry, evaluator in zip(
                self.area_geometries,
                self.area_evaluators,
            )
            if isinstance(geometry, Area)
        ]
        resolutions = self._normalize_resolutions(
            res_list,
            len(area_pairs),
            "res_list",
            allow_area_pair=True,
        )
        for (_, evaluator), resolution in zip(area_pairs, resolutions):
            evaluator.sampling_area(resolution, scheme=scheme)

    def define_time(
        self,
        range_t: Union[float, int, List[float]],
        sampling_scheme: str = "uniform",
        expo_scaling: Optional[bool] = None,
    ) -> None:
        """Define and refresh time coordinates for every child geometry."""
        for evaluator in self._ordered_evaluators:
            evaluator.define_time(
                range_t,
                sampling_scheme=sampling_scheme,
                expo_scaling=expo_scaling,
            )

    def _aggregate_plot_data(
        self,
        color_axis: str,
        x_axis: str,
        y_axis: str,
    ) -> Dict[str, np.ndarray]:
        """Build a temporary aligned dataset for an aggregate color plot."""
        datasets = []
        required_keys = (x_axis, y_axis, color_axis)

        for evaluator in self._ordered_evaluators:
            data = evaluator.data_dict
            if any(key not in data for key in required_keys):
                continue

            arrays = {
                key: np.asarray(data[key]).reshape(-1)
                for key in required_keys
            }
            lengths = {key: value.shape[0] for key, value in arrays.items()}
            if len(set(lengths.values())) != 1:
                geometry = evaluator.geometry
                details = ", ".join(
                    f"{key}={length}" for key, length in lengths.items()
                )
                raise ValueError(
                    "Cannot aggregate plot_color for "
                    f"{type(geometry).__name__}: {details}."
                )

            datasets.append(arrays)

        if not datasets:
            raise KeyError(
                "No evaluated geometry contains complete plotting data for "
                f"color_axis={color_axis!r}, x_axis={x_axis!r}, "
                f"y_axis={y_axis!r}."
            )

        return {
            key: np.concatenate([dataset[key] for dataset in datasets])
            for key in required_keys
        }

    def plot(self, *args, geometry, **kwargs):
        return self.get_evaluator(geometry).plot(*args, **kwargs)

    @staticmethod
    def _parse_plot_color_args(args, kwargs) -> Dict[str, Any]:
        """Resolve plot_color arguments for the aggregate path."""
        names = (
            "color_axis",
            "x_axis",
            "y_axis",
            "cmap",
            "s",
            "orientation",
            "return_ax",
        )
        values = {
            "color_axis": None,
            "x_axis": "x",
            "y_axis": "y",
            "cmap": "viridis",
            "s": 2,
            "orientation": "vertical",
            "return_ax": False,
        }

        if len(args) > len(names):
            raise TypeError(
                f"plot_color() takes at most {len(names)} positional "
                f"arguments ({len(args)} were given)"
            )

        provided = set()
        for name, value in zip(names, args):
            values[name] = value
            provided.add(name)

        for name, value in kwargs.items():
            if name not in values:
                raise TypeError(
                    f"plot_color() got an unexpected keyword argument {name!r}"
                )
            if name in provided:
                raise TypeError(
                    f"plot_color() got multiple values for argument {name!r}"
                )
            values[name] = value

        if values["color_axis"] is None:
            raise TypeError(
                "plot_color() missing 1 required positional argument: "
                "'color_axis'"
            )
        return values

    def plot_color(self, *args, geometry=None, **kwargs):
        if geometry is not None:
            return self.get_evaluator(geometry).plot_color(*args, **kwargs)

        plot_args = self._parse_plot_color_args(args, kwargs)
        aggregate_data = self._aggregate_plot_data(
            color_axis=plot_args["color_axis"],
            x_axis=plot_args["x_axis"],
            y_axis=plot_args["y_axis"],
        )
        return Visualizer(aggregate_data).plot_color(**plot_args)

    plot_scatter = plot_color

    def plot_contour(self, *args, geometry, **kwargs):
        return self.get_evaluator(geometry).plot_contour(*args, **kwargs)

    def plot_streamline(self, *args, geometry, **kwargs):
        return self.get_evaluator(geometry).plot_streamline(*args, **kwargs)

    def plot_distribution(self, *args, geometry, **kwargs):
        return self.get_evaluator(geometry).plot_distribution(*args, **kwargs)

    def plot_loss_curve(self, *args, geometry, **kwargs):
        return self.get_evaluator(geometry).plot_loss_curve(*args, **kwargs)

    def plot_animate(self, *args, geometry, **kwargs):
        return self.get_evaluator(geometry).plot_animate(*args, **kwargs)
