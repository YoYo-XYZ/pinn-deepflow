#!/usr/bin/env python3
"""
Compare DeepFlow Burgers-equation performance with Glorot-normal (current default)
vs the old Kaiming-uniform initialization.

The script runs a short benchmark for both initializers and prints the metrics
side by side.
"""

import sys
from pathlib import Path

from torch import pi, sin

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from benchmark_utils import (  # noqa: E402
    add_project_src,
    print_summary,
    run_comparison,
    save_field_plot,
)

add_project_src(__file__)
import deepflow as df  # noqa: E402
from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    DEPTH,
    EPOCHS,
    EVAL_GRID,
    INTERIOR_POINTS,
    LR,
    NU,
    SEED,
    WIDTH,
    X_RANGE,
    Y_RANGE,
)


def build_domain():
    area = df.geometry.rectangle(list(X_RANGE), list(Y_RANGE))
    line_ic = df.geometry.line_horizontal(y=Y_RANGE[0], range_x=list(X_RANGE))
    line_bc1 = df.geometry.line_vertical(x=X_RANGE[0], range_y=list(Y_RANGE))
    line_bc2 = df.geometry.line_vertical(x=X_RANGE[1], range_y=list(Y_RANGE))
    domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)

    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})

    domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)
    return domain


def main():
    results = run_comparison(
        df,
        build_domain,
        seed=SEED,
        input_vars=["x", "y"],
        output_vars=["u"],
        width=WIDTH,
        depth=DEPTH,
        learning_rate=LR,
        epochs=EPOCHS,
        eval_grid=EVAL_GRID,
        residual_keys={"max_pde_residual": "pde_residual"},
        field_names=["u"],
    )
    print_summary(
        results,
        [
            ("final_total", "Final total loss"),
            ("final_bc", "Final BC loss"),
            ("final_pde", "Final PDE loss"),
            ("max_pde_residual", "Max |PDE residual|"),
            ("time", "Train time (s)"),
        ],
    )
    save_field_plot(
        results,
        "u",
        Path(__file__).parent / "results" / "init_u_field.png",
        "u-field figure",
    )


if __name__ == "__main__":
    main()
