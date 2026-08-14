"""Staged smoke checks for the cylinder PINN/QCPINN benchmark."""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import build_domain, count_params, df  # noqa: E402
from benchmark_pinn_psip import build_model as build_pinn_psip  # noqa: E402
from benchmark_pinn_uvp import build_model as build_pinn_uvp  # noqa: E402
from benchmark_qcpinn_psip import build_model as build_qcpinn_psip  # noqa: E402
from benchmark_qcpinn_uvp import build_model as build_qcpinn_uvp  # noqa: E402


def _finite_loss(domain, model):
    losses = df.calc_loss_simple(domain)(model)
    value = float(losses["total_loss"].detach().cpu().item())
    assert np.isfinite(value), "loss is not finite"
    return value


def pinn_reference_smoke():
    print("PINN/reference smoke test")
    df.manual_seed(69)
    points = [4, 4, 4, 4, 4, 4]
    interior = [[4, 4]]
    domain = build_domain("uvp", points, interior)
    model = build_pinn_uvp().to(df.device)
    assert count_params(model) == 7347
    _finite_loss(domain, model)
    _, best = model.train_adam(
        calc_loss=df.calc_loss_simple(domain),
        learning_rate=0.004,
        epochs=1,
        print_every=1,
    )
    area = domain.area_list[0].evaluate(best)
    area.sampling_area([5, 5])
    assert area.data_dict["u"].size > 0

    reference_domain = build_domain("uvp", points, interior)
    reference = reference_domain.solve_fem(
        mesh_size=0.15,
        boundary_resolution=24,
        tolerance=1.0e-4,
        max_iterations=40,
    )
    assert reference.metadata["converged"]
    fields = reference.evaluate(
        area.data_dict["x"], area.data_dict["y"], fields=("u", "v", "p")
    )
    assert all(np.all(np.isfinite(np.asarray(fields[name]))) for name in ("u", "v", "p"))
    print(f"  PINN parameters: {count_params(model)}")
    print(f"  FEM iterations: {reference.metadata['iterations']}")
    print("  PINN/reference smoke test passed.")


def all_setups_smoke():
    print("All-setups smoke test")
    checks = (
        ("PINN-UVP", "uvp", build_pinn_uvp, 7347, 3),
        ("PINN-PSIP", "psip", build_pinn_psip, 7298, 2),
        ("QCPINN-UVP", "uvp", build_qcpinn_uvp, 607, 3),
        ("QCPINN-PSIP", "psip", build_qcpinn_psip, 574, 2),
    )
    for label, formulation, factory, expected_params, expected_residuals in checks:
        df.manual_seed(69)
        domain = build_domain(formulation, [4, 4, 4, 4, 4, 4], [[4, 4]])
        model = factory().to(df.device)
        assert count_params(model) == expected_params
        _finite_loss(domain, model)
        area = domain.area_list[0].evaluate(model)
        area.sampling_area([5, 5])
        assert all(area.data_dict[field].size > 0 for field in ("u", "v", "p"))
        pde_fields = (
            ("continuity_residual", "x_momentum_residual", "y_momentum_residual")
            if formulation == "uvp"
            else ("x_momentum_residual", "y_momentum_residual")
        )
        assert len(pde_fields) == expected_residuals
        assert all(field in area.data_dict for field in pde_fields)
        if formulation == "psip":
            assert np.max(np.abs(area.data_dict["continuity_residual"])) < 1.0e-5
        print(f"  {label}: {expected_params} parameters, {expected_residuals} residuals")
    print("  All-setups smoke test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-reference", action="store_true")
    parser.add_argument("--all-setups", action="store_true")
    args = parser.parse_args()
    if not args.pinn_reference and not args.all_setups:
        args.pinn_reference = args.all_setups = True
    if args.pinn_reference:
        pinn_reference_smoke()
    if args.all_setups:
        all_setups_smoke()


if __name__ == "__main__":
    main()
