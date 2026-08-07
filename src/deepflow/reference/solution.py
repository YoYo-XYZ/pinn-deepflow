"""Point-queryable and exportable reference solution objects."""

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


class ReferenceSolution:
    """A solved field collection backed by NGSolve grid functions.

    ``ReferenceSolution`` deliberately keeps the FEM objects alive.  Queries
    therefore evaluate the same fields that were produced by the solver,
    rather than interpolating a second sampled representation.
    """

    def __init__(
        self,
        *,
        area,
        mesh,
        fields: Optional[Mapping[str, object]] = None,
        snapshots: Optional[Sequence[Mapping[str, object]]] = None,
        times: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, object]] = None,
        field_evaluator: Optional[Callable[[object, np.ndarray, np.ndarray], np.ndarray]] = None,
        time_from_y: bool = False,
    ):
        if snapshots is not None and times is None:
            raise ValueError("Transient snapshots require corresponding times.")
        if snapshots is None and fields is None:
            raise ValueError("A reference solution must contain fields.")

        self._area = area
        self.mesh = mesh
        self._fields = dict(fields or {})
        self._snapshots = [dict(snapshot) for snapshot in snapshots] if snapshots is not None else None
        self._times = np.asarray(times, dtype=float) if times is not None else None
        self._metadata = dict(metadata or {})
        self._field_evaluator = field_evaluator or (
            lambda field, x, y: self._evaluate_ngsolve_field(self.mesh, field, x, y)
        )
        self._time_from_y = bool(time_from_y)
        self._query_cache: Dict[tuple, Dict[str, np.ndarray]] = {}

        if self._snapshots is not None:
            if len(self._snapshots) != len(self._times):
                raise ValueError("The number of snapshots and times must match.")
            if len(self._times) == 0 or np.any(np.diff(self._times) <= 0):
                raise ValueError("Snapshot times must be strictly increasing.")
            field_names = tuple(self._snapshots[0].keys())
        else:
            field_names = tuple(self._fields.keys())

        self.declared_fields = field_names
        self.derived_fields = tuple(self._metadata.get("derived_fields", ()))
        self._metadata.setdefault("fields", list(self.declared_fields))
        self._metadata.setdefault("derived_fields", list(self.derived_fields))
        if self._times is not None:
            self._metadata.setdefault("time_values", self._times.tolist())

    @property
    def metadata(self) -> dict:
        """Solver metadata, including mesh and convergence information."""
        return self._metadata

    @property
    def fields(self) -> Tuple[str, ...]:
        """Names of fields accepted by :meth:`evaluate`."""
        return self.declared_fields

    @property
    def times(self) -> Optional[np.ndarray]:
        """Stored transient snapshot times, or ``None`` for steady fields."""
        return None if self._times is None else self._times.copy()

    @property
    def is_transient(self) -> bool:
        return self._snapshots is not None

    @property
    def time_from_y(self) -> bool:
        """Whether the existing ``y`` coordinate carries transient time."""
        return self._time_from_y

    @staticmethod
    def _evaluate_ngsolve_field(mesh, field, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate a scalar NGSolve field point by point.

        NGSolve's point evaluator accepts one coordinate in 1-D and two in
        2-D.  The solver supplies a custom evaluator for the Burgers backend;
        this default handles all 2-D FEM fields.
        """
        values = np.empty(x.size, dtype=float)
        for index, (x_value, y_value) in enumerate(zip(x.flat, y.flat)):
            value = field(mesh(float(x_value), float(y_value)))
            values[index] = float(np.real(value))
        return values

    def _evaluate_field(self, field, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        values = np.asarray(self._field_evaluator(field, x, y), dtype=float)
        if values.shape != x.shape:
            values = np.broadcast_to(values, x.shape).astype(float, copy=False)
        return values

    def _inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_tensor = torch.as_tensor(np.array(x, copy=True), dtype=torch.float64)
        y_tensor = torch.as_tensor(np.array(y, copy=True), dtype=torch.float64)
        try:
            mask = self._area.contains(x_tensor, y_tensor)
        except Exception as exc:
            raise ValueError("Could not validate query points against the PDE area.") from exc
        inside = np.asarray(mask.detach().cpu(), dtype=bool)
        if inside.all():
            return inside

        boundary_points = []
        for bound in list(getattr(self._area, "bound_list", []) or []) + list(
            getattr(self._area, "negative_bound_list", []) or []
        ):
            bound_x = getattr(bound, "X", None)
            bound_y = getattr(bound, "Y", None)
            if bound_x is None or bound_y is None:
                continue
            bound_points = np.column_stack(
                [
                    np.asarray(bound_x.detach().cpu() if isinstance(bound_x, torch.Tensor) else bound_x),
                    np.asarray(bound_y.detach().cpu() if isinstance(bound_y, torch.Tensor) else bound_y),
                ]
            )
            if bound_points.size:
                boundary_points.append(bound_points)

        if not boundary_points:
            return inside

        query_points = np.column_stack([np.asarray(x), np.asarray(y)])
        scale = max(
            1.0,
            float(np.max(np.abs(query_points))) if query_points.size else 1.0,
        )
        tolerance = 1.0e-7 * scale
        boundary_keys = {
            tuple(key)
            for points in boundary_points
            for key in np.rint(points / tolerance).astype(np.int64)
        }
        query_keys = np.rint(query_points / tolerance).astype(np.int64)
        boundary_mask = np.asarray(
            [tuple(key) in boundary_keys for key in query_keys],
            dtype=bool,
        )
        return inside | boundary_mask

    def _select_fields(self, fields: Optional[Iterable[str]]) -> Tuple[str, ...]:
        if fields is None:
            return self.declared_fields
        if isinstance(fields, str):
            selected = (fields,)
        else:
            selected = tuple(fields)
        unknown = sorted(set(selected) - set(self.declared_fields))
        if unknown:
            raise KeyError(
                "Unknown reference field(s): " + ", ".join(unknown)
            )
        return selected

    @staticmethod
    def _array_key(array: Optional[np.ndarray]):
        if array is None:
            return None
        return (array.shape, str(array.dtype), array.tobytes())

    def _evaluate_snapshot(self, snapshot, x: np.ndarray, y: np.ndarray, fields):
        return {
            name: self._evaluate_field(snapshot[name], x, y)
            for name in fields
        }

    def evaluate(self, x, y, t=None, fields=None) -> Dict[str, np.ndarray]:
        """Evaluate selected fields at broadcastable point coordinates.

        For transient solutions, values are linearly interpolated between the
        stored solver snapshots.  A transient query must provide ``t``;
        steady fields ignore a supplied time coordinate.
        """
        x_input = np.asarray(x, dtype=float)
        y_input = np.asarray(y, dtype=float)
        if t is not None:
            x_array, y_array, t_input = np.broadcast_arrays(
                x_input, y_input, np.asarray(t, dtype=float)
            )
        else:
            x_array, y_array = np.broadcast_arrays(x_input, y_input)
            t_input = None

        if t is None and self._time_from_y:
            t_array = np.asarray(y_array, dtype=float)
            domain_y_array = y_array
            query_shape = x_array.shape
        elif t is None:
            if self.is_transient:
                raise ValueError("Transient reference queries require t.")
            t_array = None
            domain_y_array = y_array
            query_shape = x_array.shape
        else:
            t_array = t_input
            domain_y_array = t_array if self._time_from_y else y_array
            query_shape = x_array.shape

        x_flat = np.asarray(x_array, dtype=float).reshape(-1)
        y_flat = np.asarray(domain_y_array, dtype=float).reshape(-1)
        t_flat = None if t_array is None else np.asarray(t_array).reshape(-1)

        inside = self._inside(x_flat, y_flat)
        if not inside.all():
            raise ValueError("Reference query points must lie inside the PDE domain.")

        selected = self._select_fields(fields)
        key = (
            tuple(selected),
            self._array_key(x_array),
            self._array_key(y_array),
            self._array_key(t_array),
        )
        if key in self._query_cache:
            return {name: value.copy() for name, value in self._query_cache[key].items()}

        if not self.is_transient:
            values = self._evaluate_snapshot(self._fields, x_flat, y_flat, selected)
        else:
            if np.any(t_flat < self._times[0] - 1.0e-12) or np.any(
                t_flat > self._times[-1] + 1.0e-12
            ):
                raise ValueError(
                    f"Transient query times must lie in [{self._times[0]}, {self._times[-1]}]."
                )
            t_flat = np.clip(t_flat, self._times[0], self._times[-1])
            upper = np.searchsorted(self._times, t_flat, side="right")
            upper = np.clip(upper, 1, len(self._times) - 1)
            lower = upper - 1
            exact_first = t_flat <= self._times[0]
            exact_last = t_flat >= self._times[-1]

            values = {name: np.empty(x_flat.shape, dtype=float) for name in selected}
            for lower_index in np.unique(lower):
                mask = lower == lower_index
                first_values = self._evaluate_snapshot(
                    self._snapshots[lower_index], x_flat[mask], y_flat[mask], selected
                )
                second_values = self._evaluate_snapshot(
                    self._snapshots[lower_index + 1], x_flat[mask], y_flat[mask], selected
                )
                fraction = (
                    (t_flat[mask] - self._times[lower_index])
                    / (self._times[lower_index + 1] - self._times[lower_index])
                )
                for name in selected:
                    values[name][mask] = first_values[name] + fraction * (
                        second_values[name] - first_values[name]
                    )

            if exact_first.any():
                first_values = self._evaluate_snapshot(
                    self._snapshots[0], x_flat[exact_first], y_flat[exact_first], selected
                )
                for name in selected:
                    values[name][exact_first] = first_values[name]
            if exact_last.any():
                last_values = self._evaluate_snapshot(
                    self._snapshots[-1], x_flat[exact_last], y_flat[exact_last], selected
                )
                for name in selected:
                    values[name][exact_last] = last_values[name]

        values = {
            name: np.asarray(value, dtype=float).reshape(query_shape)
            for name, value in values.items()
        }
        self._query_cache[key] = {name: value.copy() for name, value in values.items()}
        return values

    def export_npz(self, path, x, y, t=None, fields=None) -> None:
        """Export queried values and JSON-serializable metadata to ``.npz``."""
        values = self.evaluate(x, y, t=t, fields=fields)
        if t is None:
            x_array, y_array = np.broadcast_arrays(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            )
        else:
            x_array, y_array, t_array = np.broadcast_arrays(
                np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(t, dtype=float)
            )
        payload = {"x": x_array, "y": y_array}
        if t is not None:
            payload["t"] = t_array
        if self._times is not None:
            payload["time_values"] = self._times
        payload.update(values)
        payload["metadata"] = np.asarray(
            json.dumps(self._metadata, default=str), dtype=str
        )
        np.savez(Path(path), **payload)
