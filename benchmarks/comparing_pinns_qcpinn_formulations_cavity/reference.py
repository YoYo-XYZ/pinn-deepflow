"""Canonical FEM reference and explicit offline-cache support."""

from __future__ import annotations

import argparse
from pathlib import Path

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

from benchmarks.shared_harness.reference import (
    CachedReference,
    export_reference_cache,
    load_cached_reference,
)


def solve_reference(
    config=DEFAULT_CONFIG,
    *,
    mesh_size: float = FEM_MESH_SIZE,
    boundary_resolution: int = FEM_BOUNDARY_RESOLUTION,
    tolerance: float = FEM_TOLERANCE,
    max_iterations: int = FEM_MAX_ITERATIONS,
    output_path: Path | None = None,
):
    """Solve the cavity with DeepFlow's canonical FEM entry point."""
    reference_domain = build_domain("uvp", config)
    reference = reference_domain.solve_fem(
        mesh_size=mesh_size,
        boundary_resolution=boundary_resolution,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if not reference.metadata.get("converged", True):
        raise RuntimeError("DeepFlow FEM cavity reference did not converge.")
    if output_path is not None:
        export_reference_cache(reference, output_path, config.eval_grid)
    return reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-cache",
        type=Path,
        help="Explicitly export queried FEM values for later offline comparison.",
    )
    args = parser.parse_args(argv)
    reference = solve_reference(DEFAULT_CONFIG, output_path=args.export_cache)
    print("FEM cavity reference converged")
    if args.export_cache:
        print(f"Offline cache: {args.export_cache}")
    return reference


if __name__ == "__main__":
    main()
