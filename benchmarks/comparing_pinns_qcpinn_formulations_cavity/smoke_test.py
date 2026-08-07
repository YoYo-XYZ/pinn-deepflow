"""Small regression test for all four cavity benchmark setups."""

import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from benchmark_common import build_domain, count_params, df  # noqa: E402
from benchmark_pinn_psip import build_model as build_pinn_psip  # noqa: E402
from benchmark_pinn_uvp import build_model as build_pinn_uvp  # noqa: E402
from benchmark_qcpinn_psip import build_model as build_qcpinn_psip  # noqa: E402
from benchmark_qcpinn_uvp import build_model as build_qcpinn_uvp  # noqa: E402


def check_setup(label, formulation, model_factory, expected_params, expected_residuals):
    df.manual_seed(69)
    domain = build_domain(
        formulation,
        boundary_points=[4, 4, 4, 4, 1],
        interior_points=[[4, 4]],
    )
    model = model_factory().to(df.device)
    assert count_params(model) == expected_params

    loss = df.calc_loss_simple(domain)(model)
    assert len(domain.area_list[0].PDE.residual_fields) == expected_residuals
    assert np.isfinite(loss["total_loss"].detach().cpu().item())

    area_eval = domain.area_list[0].evaluate(model)
    area_eval.sampling_area([5, 5])
    for field in ("u", "v", "p"):
        assert area_eval.data_dict[field].size == 25
    if formulation == "psip":
        assert area_eval.data_dict["psi"].size == 25
        assert np.max(np.abs(area_eval.data_dict["continuity_residual"])) < 1.0e-5
    print(f"{label}: {expected_params} parameters, {expected_residuals} PDE residuals")


check_setup("PINN-UVP", "uvp", build_pinn_uvp, 7347, 3)
check_setup("PINN-PSIP", "psip", build_pinn_psip, 7298, 2)
check_setup("QCPINN-UVP", "uvp", build_qcpinn_uvp, 607, 3)
check_setup("QCPINN-PSIP", "psip", build_qcpinn_psip, 574, 2)

print("All four cavity benchmark setups passed the smoke test.")
