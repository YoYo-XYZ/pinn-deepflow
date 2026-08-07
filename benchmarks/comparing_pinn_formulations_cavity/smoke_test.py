"""Small regression test for both cavity PDE formulations."""

import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from benchmark_common import (  # noqa: E402
    build_domain,
    count_params,
    df,
)
from benchmark_psip import build_model as build_psip_model  # noqa: E402
from benchmark_uvp import build_model as build_uvp_model  # noqa: E402


def check_formulation(formulation, model_factory, expected_params, expected_residuals):
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
    assert area_eval.data_dict["u"].size == 25
    assert area_eval.data_dict["v"].size == 25
    assert area_eval.data_dict["p"].size == 25
    return domain, area_eval


uvp_domain, _ = check_formulation("uvp", build_uvp_model, 7347, 3)
psip_domain, psip_eval = check_formulation("psip", build_psip_model, 7349, 2)

assert uvp_domain.bound_list[0].condition_dict == {"u": 0, "v": 0}
assert psip_domain.bound_list[0].condition_dict == {"psi_x": 0, "psi_y": 0}
assert psip_domain.bound_list[3].condition_dict == {"psi_x": 0, "psi_y": 1.0}
assert psip_domain.bound_list[4].condition_dict == {"p": 0}
assert np.max(np.abs(psip_eval.data_dict["continuity_residual"])) < 1.0e-5

print("UVP and PSIP cavity geometry, residuals, parameters, and evaluation OK.")
