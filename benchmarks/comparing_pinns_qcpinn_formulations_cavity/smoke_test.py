"""Run the shared-harness smoke path for the cavity benchmark."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

try:  # Package execution.
    from .benchmark import (  # noqa: E402
        SMOKE_CONFIG,
        available_variants,
        run_suite,
    )
    from .reference import solve_reference  # noqa: E402
except ImportError:  # Direct script execution.
    from benchmark import (  # type: ignore  # noqa: E402
        SMOKE_CONFIG,
        available_variants,
        run_suite,
    )
    from reference import solve_reference  # type: ignore  # noqa: E402


def _assert_suite(result):
    assert result["report"].exists()
    for variant, values in result["variants"].items():
        assert values["evaluator"].data_dict["u"].size > 0
        assert values["profiles"]["vertical"].data_dict["u"].size > 0
        assert values["profiles"]["horizontal"].data_dict["v"].size > 0
        assert np.isfinite(values["metrics"]["final_total_loss"])
        assert all(path.exists() for path in values["artifacts"])
        print(f"  {variant}: shared-harness smoke passed")


def all_setups_smoke(output_dir: Path):
    result = run_suite(
        SMOKE_CONFIG,
        output_dir,
        variants=available_variants(),
    )
    _assert_suite(result)
    return result


def pinn_reference_smoke():
    """Exercise the canonical FEM/reference seam when the backend exists."""
    try:
        reference = solve_reference(SMOKE_CONFIG)
    except (ImportError, OSError) as exc:
        print(f"  FEM reference skipped: {exc}")
        return None
    with tempfile.TemporaryDirectory(prefix="deepflow-cavity-reference-smoke-") as directory:
        result = run_suite(
            SMOKE_CONFIG,
            Path(directory),
            variants=("PINN-UVP",),
            reference_solution=reference,
        )
        _assert_suite(result)
        assert "relative_l2" in result["variants"]["PINN-UVP"]["metrics"]
        return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-reference", action="store_true")
    parser.add_argument("--all-setups", action="store_true")
    args = parser.parse_args(argv)
    if not args.pinn_reference and not args.all_setups:
        args.pinn_reference = args.all_setups = True
    if args.pinn_reference:
        pinn_reference_smoke()
    if args.all_setups:
        with tempfile.TemporaryDirectory(prefix="deepflow-cavity-smoke-") as directory:
            all_setups_smoke(Path(directory))


if __name__ == "__main__":
    main()
