import operator
from typing import Callable, List, Union, Dict, Any, Optional

import torch
import numpy as np
import ultraplot as plt

# Explicit imports are better for open source
from .nn import PINN 
from .geometry import Area, Bound, CustomData
from .visualization import Visualizer


def _unique_geometry_entries(geometries):
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


def _validate_sampled_entries(bound_entries, area_entries, operation: str) -> None:
    """Raise a focused error when an included geometry lacks coordinates."""
    missing = []
    for category, entries in (("bound", bound_entries), ("area", area_entries)):
        for index, geometry in entries:
            if (
                getattr(geometry, "X", None) is None
                or getattr(geometry, "Y", None) is None
            ):
                missing.append(f"{category}[{index}] ({type(geometry).__name__})")

    if missing:
        raise ValueError(
            "Cannot evaluate unsampled domain geometries: "
            f"{', '.join(missing)}. Sample every geometry before calling "
            f"{operation}()."
        )


def _normalize_resolutions(
    resolution,
    count: int,
    name: str,
    *,
    allow_area_pair: bool = False,
) -> list:
    """Normalize scalar or aligned resolution values for child geometries."""
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


class _DerivedExpressionCycleError(ValueError):
    """Internal marker used to preserve cycle diagnostics."""


class _LazyFieldExpression:
    """Lazy expression tree used by :attr:`Evaluator.expr`."""

    __array_priority__ = 1000

    def __init__(
        self,
        evaluator,
        kind: str,
        value=None,
        args=(),
        kwargs=None,
        text: Optional[str] = None,
    ) -> None:
        self._evaluator = evaluator
        self._kind = kind
        self._value = value
        self._args = tuple(args)
        self._kwargs = dict(kwargs or {})
        self._text = text

    @classmethod
    def field(cls, evaluator, name: str):
        return cls(evaluator, "field", value=name, text=name)

    @classmethod
    def literal(cls, evaluator, value):
        return cls(evaluator, "literal", value=value, text=repr(value))

    @classmethod
    def call(cls, evaluator, function, args, kwargs, text: str):
        normalized_args = tuple(
            cls._coerce(evaluator, value) for value in args
        )
        normalized_kwargs = {
            key: cls._coerce(evaluator, value)
            for key, value in kwargs.items()
        }
        return cls(
            evaluator,
            "call",
            value=function,
            args=normalized_args,
            kwargs=normalized_kwargs,
            text=text,
        )

    @classmethod
    def _coerce(cls, evaluator, value):
        if isinstance(value, cls):
            if value._evaluator is not evaluator:
                raise ValueError(
                    "Cannot combine expressions belonging to different "
                    "Evaluator instances."
                )
            return value
        return cls.literal(evaluator, value)

    def _binary(self, other, function: Callable, symbol: str, reverse=False):
        other = self._coerce(self._evaluator, other)
        left, right = (other, self) if reverse else (self, other)
        return self.call(
            self._evaluator,
            function,
            (left, right),
            {},
            f"({left!r} {symbol} {right!r})",
        )

    def _unary(self, function: Callable, symbol: str):
        return self.call(
            self._evaluator,
            function,
            (self,),
            {},
            f"({symbol}{self!r})",
        )

    def __add__(self, other):
        return self._binary(other, operator.add, "+")

    def __radd__(self, other):
        return self._binary(other, operator.add, "+", reverse=True)

    def __sub__(self, other):
        return self._binary(other, operator.sub, "-")

    def __rsub__(self, other):
        return self._binary(other, operator.sub, "-", reverse=True)

    def __mul__(self, other):
        return self._binary(other, operator.mul, "*")

    def __rmul__(self, other):
        return self._binary(other, operator.mul, "*", reverse=True)

    def __truediv__(self, other):
        return self._binary(other, operator.truediv, "/")

    def __rtruediv__(self, other):
        return self._binary(other, operator.truediv, "/", reverse=True)

    def __floordiv__(self, other):
        return self._binary(other, operator.floordiv, "//")

    def __rfloordiv__(self, other):
        return self._binary(other, operator.floordiv, "//", reverse=True)

    def __mod__(self, other):
        return self._binary(other, operator.mod, "%")

    def __rmod__(self, other):
        return self._binary(other, operator.mod, "%", reverse=True)

    def __pow__(self, other):
        return self._binary(other, operator.pow, "**")

    def __rpow__(self, other):
        return self._binary(other, operator.pow, "**", reverse=True)

    def __neg__(self):
        return self._unary(operator.neg, "-")

    def __pos__(self):
        return self._unary(operator.pos, "+")

    def __abs__(self):
        return self.call(
            self._evaluator,
            operator.abs,
            (self,),
            {},
            f"abs({self!r})",
        )

    def __lt__(self, other):
        return self._binary(other, operator.lt, "<")

    def __le__(self, other):
        return self._binary(other, operator.le, "<=")

    def __gt__(self, other):
        return self._binary(other, operator.gt, ">")

    def __ge__(self, other):
        return self._binary(other, operator.ge, ">=")

    def __eq__(self, other):
        return self._binary(other, operator.eq, "==")

    def __ne__(self, other):
        return self._binary(other, operator.ne, "!=")

    def __bool__(self):
        raise TypeError(
            "A lazy field expression cannot be used as a Python boolean. "
            "Use np.where() for element-wise conditions."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise TypeError(
                f"NumPy ufunc method {method!r} is not supported for lazy "
                "field expressions."
            )
        if kwargs.get("out") is not None:
            raise TypeError("Lazy field expressions do not support NumPy 'out'.")

        text = f"{ufunc.__name__}({', '.join(repr(value) for value in inputs)})"
        return self.call(
            self._evaluator,
            ufunc,
            inputs,
            kwargs,
            text,
        )

    def __array_function__(self, function, types, args, kwargs):
        if function is not np.where:
            raise TypeError(
                f"NumPy function {function.__name__!r} is not supported for "
                "lazy field expressions."
            )

        text = f"where({', '.join(repr(value) for value in args)})"
        return self.call(
            self._evaluator,
            np.where,
            args,
            kwargs,
            text,
        )

    def _evaluate(self, resolve: Callable[[str], Any]):
        if self._kind == "field":
            return resolve(self._value)
        if self._kind == "literal":
            return self._value

        args = tuple(value._evaluate(resolve) for value in self._args)
        kwargs = {
            key: value._evaluate(resolve)
            for key, value in self._kwargs.items()
        }
        return self._value(*args, **kwargs)

    def __repr__(self) -> str:
        return self._text or "<lazy field expression>"


class _ExpressionNamespace:
    """Mapping-like namespace for defining persistent field expressions."""

    def __init__(self, evaluator) -> None:
        self._evaluator = evaluator

    def __getitem__(self, key: str) -> _LazyFieldExpression:
        if not isinstance(key, str):
            raise TypeError("Expression field names must be strings.")
        return _LazyFieldExpression.field(self._evaluator, key)

    def __setitem__(self, key: str, expression: _LazyFieldExpression) -> None:
        if not isinstance(key, str):
            raise TypeError("Expression field names must be strings.")
        if not isinstance(expression, _LazyFieldExpression):
            raise TypeError(
                "Evaluator.expr values must be lazy field expressions."
            )
        if expression._evaluator is not self._evaluator:
            raise ValueError(
                "Cannot assign an expression belonging to a different "
                "Evaluator instance."
            )
        self._evaluator._set_derived_expression(key, expression)

    def __delitem__(self, key: str) -> None:
        self._evaluator._remove_derived_expression(key)

    def __repr__(self) -> str:
        names = tuple(self._evaluator._derived_expressions)
        return f"ExpressionNamespace(derived_fields={names})"


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
        self._derived_expressions: Dict[str, _LazyFieldExpression] = {}
        self._expression_namespace = _ExpressionNamespace(self)
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
        self._refresh_derived_fields()
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

    @property
    def expr(self) -> _ExpressionNamespace:
        """Namespace for defining persistent lazy data-field expressions."""
        return self._expression_namespace

    def _evaluate_derived_fields(self) -> Dict[str, np.ndarray]:
        """Evaluate all registered expressions against the current data."""
        values = {}
        memo = {}
        visiting = []

        def resolve(key: str):
            if key in memo:
                return memo[key]
            if key in self._derived_expressions:
                if key in visiting:
                    cycle = " -> ".join(visiting + [key])
                    raise _DerivedExpressionCycleError(
                        f"Cyclic derived-field dependency detected: {cycle}."
                    )

                visiting.append(key)
                try:
                    value = self._derived_expressions[key]._evaluate(resolve)
                except KeyError as exc:
                    missing = exc.args[0] if exc.args else "unknown"
                    raise KeyError(
                        f"Cannot evaluate derived field {key!r}: missing "
                        f"data key {missing!r}."
                    ) from exc
                except _DerivedExpressionCycleError:
                    raise
                except Exception as exc:
                    raise ValueError(
                        f"Failed to evaluate derived field {key!r} "
                        f"({self._derived_expressions[key]!r})."
                    ) from exc
                finally:
                    visiting.pop()

                memo[key] = np.asarray(value)
                return memo[key]

            if key not in self.data_dict:
                raise KeyError(key)
            return self.data_dict[key]

        for key in self._derived_expressions:
            values[key] = resolve(key)
        return values

    def _refresh_derived_fields(self) -> None:
        """Recompute registered expressions and update ``data_dict`` atomically."""
        values = self._evaluate_derived_fields()
        self.data_dict.update(values)

    def _set_derived_expression(
        self,
        key: str,
        expression: _LazyFieldExpression,
    ) -> None:
        if key in self.data_dict and key not in self._derived_expressions:
            raise ValueError(
                f"Cannot define derived field {key!r}: it is already a "
                "base data key."
            )

        sentinel = object()
        previous = self._derived_expressions.get(key, sentinel)
        self._derived_expressions[key] = expression
        if not self.is_postprocessed:
            return

        try:
            self._refresh_derived_fields()
        except Exception:
            if previous is sentinel:
                del self._derived_expressions[key]
            else:
                self._derived_expressions[key] = previous
            raise

    def _remove_derived_expression(self, key: str) -> None:
        if key not in self._derived_expressions:
            raise KeyError(key)

        previous = self._derived_expressions.pop(key)
        previous_names = set(self._derived_expressions) | {key}
        try:
            values = self._evaluate_derived_fields()
        except Exception:
            self._derived_expressions[key] = previous
            raise

        for name in previous_names:
            self.data_dict.pop(name, None)
        self.data_dict.update(values)
    
    def __getitem__(self, key: str) -> Any:
        return self.data_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._derived_expressions.pop(key, None)
        self.data_dict[key] = value
        if self.is_postprocessed and self._derived_expressions:
            self._refresh_derived_fields()

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


class ReferenceEvaluator(Evaluator):
    """Evaluate a solved :class:`ReferenceSolution` on one geometry.

    Reference evaluators deliberately do not calculate PINN residuals or copy
    model training history.  They retain the FEM solution and re-query it
    whenever the geometry is sampled again or its time coordinates change.
    """

    def __init__(self, reference_solution, geometry: Area | Bound | CustomData) -> None:
        self.model = None
        self.reference_solution = reference_solution
        self.geometry = geometry
        self.data_dict: Dict[str, Any] = {}
        self._derived_expressions: Dict[str, _LazyFieldExpression] = {}
        self._expression_namespace = _ExpressionNamespace(self)
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
        """Process only when the public coordinates changed.

        ``PhysicsAttach.process_coordinates`` regenerates time samples when a
        ``range_t`` is present.  Avoiding an unconditional call preserves
        existing time coordinates when ``solve_fem`` is asked to reuse them.
        """
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
        """Process coordinates without replacing already aligned time data."""
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
        """Update time coordinates and immediately re-query the FEM fields."""
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
        self._create_data_dict()
        self._refresh_derived_fields()
        self.is_postprocessed = True
        Visualizer.__init__(self, self.data_dict)


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
        self._init_entries(domain)
        self._build_children(domain, Evaluator, pinns_model)
        self._validate_sampled()

    def _init_entries(self, domain) -> None:
        self._area_entries = _unique_geometry_entries(
            getattr(domain, "area_list", [])
        )
        self._bound_entries = _unique_geometry_entries(
            getattr(domain, "bound_list", [])
        )

    def _build_children(self, domain, evaluator_type, source) -> None:
        self.area_list = [
            evaluator_type(source, geometry)
            for _, geometry in self._area_entries
        ]
        self.bound_list = [
            evaluator_type(source, geometry)
            for _, geometry in self._bound_entries
        ]
        self._ordered_evaluators = [
            *self.bound_list,
            *self.area_list,
        ]

    def _validate_sampled(self) -> None:
        """Raise a clear error when any included geometry lacks coordinates."""
        _validate_sampled_entries(
            self._bound_entries,
            self._area_entries,
            "evaluate",
        )

    def __iter__(self):
        """Iterate in the same bound-first order as ``ProblemDomain``."""
        return iter(self._ordered_evaluators)

    def __len__(self) -> int:
        return len(self._ordered_evaluators)

    def __str__(self) -> str:
        return (
            "GroupEvaluator("
            f"bound_list={len(self.bound_list)}, "
            f"area_list={len(self.area_list)})"
        )

    def postprocess(self) -> None:
        """Refresh all child evaluators after sampling or model changes."""
        self._validate_sampled()
        for evaluator in self._ordered_evaluators:
            evaluator.postprocess()

    def refresh(self) -> None:
        """Alias for :meth:`postprocess`."""
        self.postprocess()

    def sampling_line(self, n_points, scheme: str = "uniform") -> None:
        """Sample and refresh every Bound child."""
        resolutions = _normalize_resolutions(
            n_points,
            len(self.bound_list),
            "n_points",
        )
        for evaluator, resolution in zip(self.bound_list, resolutions):
            evaluator.sampling_line(resolution, scheme=scheme)

    def sampling_area(self, res_list, scheme: str = "uniform") -> None:
        """
        Sample and refresh every Area child.

        ``CustomData`` entries are intentionally left unchanged because they
        represent fixed user-provided coordinates rather than a sampler.
        """
        sampleable_areas = [
            evaluator
            for evaluator in self.area_list
            if isinstance(evaluator.geometry, Area)
        ]
        resolutions = _normalize_resolutions(
            res_list,
            len(sampleable_areas),
            "res_list",
            allow_area_pair=True,
        )
        for evaluator, resolution in zip(sampleable_areas, resolutions):
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

    def _evaluator_for_geometry(self, geometry) -> Evaluator:
        for evaluator in self._ordered_evaluators:
            if evaluator.geometry is geometry:
                return evaluator
        raise KeyError("Geometry is not part of this GroupEvaluator.")

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

    @staticmethod
    def _parse_plot_color_args(args, kwargs) -> Dict[str, Any]:
        """Resolve plot_color arguments for the aggregate path."""
        names = (
            "color_axis",
            "x_axis",
            "y_axis",
            "cmap",
            "s",
            "return_ax",
        )
        values = {
            "color_axis": None,
            "x_axis": "x",
            "y_axis": "y",
            "cmap": "viridis",
            "s": 2,
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

    def plot(self, *args, geometry, **kwargs):
        return self._evaluator_for_geometry(geometry).plot(*args, **kwargs)

    def plot_color(self, *args, geometry=None, **kwargs):
        if geometry is not None:
            return self._evaluator_for_geometry(geometry).plot_color(
                *args,
                **kwargs,
            )

        plot_args = self._parse_plot_color_args(args, kwargs)
        aggregate_data = self._aggregate_plot_data(
            color_axis=plot_args["color_axis"],
            x_axis=plot_args["x_axis"],
            y_axis=plot_args["y_axis"],
        )
        return Visualizer(aggregate_data).plot_color(**plot_args)

    plot_scatter = plot_color

    def plot_contour(self, *args, geometry, **kwargs):
        return self._evaluator_for_geometry(geometry).plot_contour(
            *args,
            **kwargs,
        )

    def plot_streamline(self, *args, geometry, **kwargs):
        return self._evaluator_for_geometry(geometry).plot_streamline(
            *args,
            **kwargs,
        )

    def plot_distribution(self, *args, geometry, **kwargs):
        return self._evaluator_for_geometry(geometry).plot_distribution(
            *args,
            **kwargs,
        )

    def plot_loss_curve(self, *args, geometry, **kwargs):
        return self._evaluator_for_geometry(geometry).plot_loss_curve(
            *args,
            **kwargs,
        )

    def plot_animate(self, *args, geometry, **kwargs):
        return self._evaluator_for_geometry(geometry).plot_animate(
            *args,
            **kwargs,
        )


class ReferenceGroupEvaluator(GroupEvaluator):
    """GroupEvaluator whose children query a solved FEM reference solution.

    This is the ``solve_fem`` result: a real :class:`GroupEvaluator` where each
    per-geometry :class:`ReferenceEvaluator` re-queries the FEM fields on its
    geometry.  The underlying :class:`ReferenceSolution` remains available as
    ``.reference_solution`` for arbitrary point queries and export.
    """

    def __init__(self, reference_solution, domain) -> None:
        self.reference_solution = reference_solution
        self.model = None
        self.domain = domain
        self.metadata = reference_solution.metadata
        self._init_entries(domain)
        self._build_children(domain, ReferenceEvaluator, reference_solution)
