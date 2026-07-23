#!/usr/bin/env python3
"""
Compare DeepFlow channel-flow performance with Kaiming-uniform (DeepFlow's
current default) and Glorot-normal (the alternative).

This is a standalone, benchmark-specific comparison. It uses the explicit
perimeter-weighted random sampling protocol in ``common_config.py`` and is not
an exact reproduction of the quickstart notebook.
"""

import sys
from pathlib import Path

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
    BENCHMARK_METADATA,
    DEPTH,
    EPOCHS,
    EVAL_GRID,
    INTERIOR_POINTS,
    Lx,
    Ly,
    LR,
    SEED,
    WIDTH,
)


def build_domain():
    rect = df.geometry.rectangle([0, Lx], [0, Ly])
    domain = df.domain(rect)
    domain.bound_list[0].define_bc({"u": 1, "v": 0})
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.bound_list[3].define_bc({"u": 0, "v": 0})
    domain.area_list[0].define_pde(
        df.pde.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000)
    )
    perimeter = 2 * (Lx + Ly)
    total_boundary = sum(BOUNDARY_POINTS)
    bound_counts = [
        int(total_boundary * Ly / perimeter),
        int(total_boundary * Lx / perimeter),
        int(total_boundary * Ly / perimeter),
        int(total_boundary * Lx / perimeter),
    ]
    domain.sampling_random(bound_counts, [INTERIOR_POINTS])
    return domain


def main():
    results = run_comparison(
        df,
        build_domain,
        seed=SEED,
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        width=WIDTH,
        depth=DEPTH,
        learning_rate=LR,
        epochs=EPOCHS,
        eval_grid=EVAL_GRID,
        residual_keys={
            "max_cont": "continuity_residual",
            "max_xmom": "x_momentum_residual",
            "max_ymom": "y_momentum_residual",
        },
        field_names=["u", "v", "p"],
    )
    print_summary(
        results,
        [
            ("final_total", "Final total loss"),
            ("final_bc", "Final BC loss"),
            ("final_pde", "Final PDE loss"),
            ("max_cont", "Max |continuity|"),
            ("max_xmom", "Max |x-momentum|"),
            ("max_ymom", "Max |y-momentum|"),
            ("time", "Train time (s)"),
        ],
        metadata=BENCHMARK_METADATA,
    )
    save_field_plot(
        results,
        "u",
        Path(__file__).parent / "results" / "init_velocity_field.png",
        "u-velocity field",
    )


if __name__ == "__main__":
    main()
