"""Canonical FEM reference and explicit offline-cache support."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

try:  # Package execution.
    from .benchmark import (  # noqa: E402
        DEFAULT_CONFIG,
        FEM_BOUNDARY_RESOLUTION,
        FEM_MAX_ITERATIONS,
        FEM_MESH_SIZE,
        FEM_TOLERANCE,
        build_domain,
    )
except ImportError:  # Direct script execution.
    from benchmark import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG,
        FEM_BOUNDARY_RESOLUTION,
        FEM_MAX_ITERATIONS,
        FEM_MESH_SIZE,
        FEM_TOLERANCE,
        build_domain,
    )


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
            self._x = np.asarray(archive["x"], dtype=float).reshape(-1)
            self._y = np.asarray(archive["y"], dtype=float).reshape(-1)
            self._values = {
                name: np.asarray(archive[name], dtype=float).reshape(-1)
                for name in self.fields
            }
        if self._x.shape != self._y.shape:
            raise ValueError("Reference cache x and y arrays must have matching shapes.")
        self.metadata = {"backend": "explicit_offline_cache", "path": str(path)}
        self._interpolators = {}
        points = np.column_stack((self._x, self._y))
        for name, values in self._values.items():
            valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
            if not valid.any():
                raise ValueError(f"Reference cache contains no finite {name!r} values.")
            # FLEX: this interpolation is used only by the explicit offline fallback.
            self._interpolators[name] = (
                LinearNDInterpolator(points[valid], values[valid], fill_value=np.nan),
                NearestNDInterpolator(points[valid], values[valid]),
            )

    def evaluate(self, x, y, t=None, fields=None):
        """Evaluate cached fields at broadcastable coordinates."""
        del t
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        selected = self.fields if fields is None else ((fields,) if isinstance(fields, str) else tuple(fields))
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


def solve_reference(
    config=DEFAULT_CONFIG,
    *,
    mesh_size: float = FEM_MESH_SIZE,
    boundary_resolution: int = FEM_BOUNDARY_RESOLUTION,
    tolerance: float = FEM_TOLERANCE,
    max_iterations: int = FEM_MAX_ITERATIONS,
    output_path: Path | None = None,
):
    """Solve the cylinder domain with DeepFlow's canonical FEM entry point."""
    reference_domain = build_domain("uvp", config)
    reference = reference_domain.solve_fem(
        mesh_size=mesh_size,
        boundary_resolution=boundary_resolution,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not reference.metadata.get("converged", True):
        raise RuntimeError("DeepFlow FEM cylinder reference did not converge.")
    if output_path is not None:
        export_reference_cache(reference, output_path, config.eval_grid)
    return reference


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


def load_cached_reference(path: Path) -> CachedReference:
    """Load a cache only when the caller explicitly selects offline mode."""
    return CachedReference(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-cache",
        type=Path,
        help="Explicitly export queried FEM values for later offline comparison.",
    )
    args = parser.parse_args(argv)
    reference = solve_reference(
        DEFAULT_CONFIG,
        output_path=args.export_cache,
    )
    print("FEM cylinder reference converged")
    if args.export_cache:
        print(f"Offline cache: {args.export_cache}")
    return reference


if __name__ == "__main__":
    main()
