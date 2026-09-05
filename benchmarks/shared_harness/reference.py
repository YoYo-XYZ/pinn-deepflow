"""Shared reference persistence and explicit offline-cache evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class CachedReference:
    """Interpolate an explicitly selected reference archive for offline use."""

    fields = ("u", "v", "p")

    def __init__(self, path: Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Reference cache not found: {path}")
        with np.load(path) as archive:
            missing = [
                name for name in ("x", "y", *self.fields) if name not in archive.files
            ]
            if missing:
                raise ValueError(f"Reference cache is missing {missing}")
            x = np.asarray(archive["x"], dtype=float)
            y = np.asarray(archive["y"], dtype=float)
            raw_values = {
                name: np.asarray(archive[name], dtype=float)
                for name in self.fields
            }

        self._points, self._values = self._normalize_grid(x, y, raw_values)
        self.metadata = {"backend": "explicit_offline_cache", "path": str(path)}
        self._interpolators = {}
        for name, values in self._values.items():
            valid = np.isfinite(self._points).all(axis=1) & np.isfinite(values)
            if not valid.any():
                raise ValueError(f"Reference cache contains no finite {name!r} values.")
            # FLEX: interpolation is used only by the explicit offline fallback.
            self._interpolators[name] = (
                LinearNDInterpolator(
                    self._points[valid], values[valid], fill_value=np.nan
                ),
                NearestNDInterpolator(self._points[valid], values[valid]),
            )

    @staticmethod
    def _normalize_grid(x, y, raw_values):
        """Normalize point exports and rectangular archives selected offline."""
        if x.shape == y.shape and all(
            raw.size == x.size for raw in raw_values.values()
        ):
            points = np.column_stack((x.reshape(-1), y.reshape(-1)))
            values = {}
            for name, raw in raw_values.items():
                if raw.size != points.shape[0]:
                    raise ValueError(
                        f"Reference cache field {name!r} has an invalid shape."
                    )
                values[name] = raw.reshape(-1)
            return points, values

        if x.ndim == 1 and y.ndim == 1:
            grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
            points = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
            values = {}
            for name, raw in raw_values.items():
                if raw.shape == grid_x.shape:
                    values[name] = raw.reshape(-1)
                elif raw.size == points.shape[0]:
                    values[name] = raw.reshape(-1)
                else:
                    raise ValueError(
                        f"Reference cache field {name!r} has an invalid shape."
                    )
            return points, values

        raise ValueError("Reference cache x and y arrays must define a common grid.")

    def evaluate(self, x, y, t=None, fields=None):
        """Evaluate cached fields at broadcastable coordinates."""
        del t
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        selected = (
            self.fields
            if fields is None
            else ((fields,) if isinstance(fields, str) else tuple(fields))
        )
        unknown = sorted(set(selected) - set(self.fields))
        if unknown:
            raise KeyError("Unknown reference field(s): " + ", ".join(unknown))
        points = np.column_stack((x_array.reshape(-1), y_array.reshape(-1)))
        values = {}
        for name in selected:
            linear, nearest = self._interpolators[name]
            result = np.asarray(linear(points), dtype=float).reshape(-1)
            missing = ~np.isfinite(result)
            if missing.any():
                result[missing] = np.asarray(nearest(points[missing]), dtype=float)
            values[name] = result.reshape(x_array.shape)
        return values


def load_cached_reference(path: Path) -> CachedReference:
    """Load a cache only when the caller explicitly selects offline mode."""
    return CachedReference(path)


def export_reference_cache(reference, path: Path, eval_grid) -> Path:
    """Export queried FEM values for an explicitly requested offline fallback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    area = reference.domain.area_list[0]
    area.sampling_area(list(eval_grid))
    reference.reference_solution.export_npz(
        path,
        area.X,
        area.Y,
        fields=("u", "v", "p"),
    )
    return path
