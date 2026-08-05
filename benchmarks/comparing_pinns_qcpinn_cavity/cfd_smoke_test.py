#!/usr/bin/env python3
"""Small regression test for the finite-volume cavity reference solver."""

import numpy as np

from reference_cfd import StaggeredSimpleSolver


def main():
    solver = StaggeredSimpleSolver(
        21,
        21,
        max_iterations=2000,
        tolerance=1.0e-7,
        report_every=1000,
    )
    result = solver.solve()
    history = result["solver_history"]

    assert result["converged"] == 1
    assert result["iterations"] > 1
    assert np.all(np.isfinite(result["u"]))
    assert np.all(np.isfinite(result["v"]))
    assert np.all(np.isfinite(result["p"]))
    assert np.allclose(solver.u[:, 0], 0.0)
    assert np.allclose(solver.u[:, -1], 0.0)
    assert np.allclose(solver.v[0, :], 0.0)
    assert np.allclose(solver.v[-1, :], 0.0)
    assert abs(result["p"][0, 0]) < 1.0e-12
    assert np.max(np.abs(result["continuity_residual"])) < 1.0e-8
    assert np.max(result["u"][-1, :]) > 0.5
    assert history[-1, 1] < history[0, 1]
    print(
        "CFD cavity smoke test passed: "
        f"{result['iterations']} iterations, residual={result['final_residual']:.3e}"
    )


if __name__ == "__main__":
    main()
