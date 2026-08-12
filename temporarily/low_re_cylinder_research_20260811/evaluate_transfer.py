"""Evaluate the shipped mu=0.02 model under pressure/viscosity transfer tests."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import deepflow as df  # noqa: E402
from experiment import ScaledPINN, build_domain, evaluate  # noqa: E402


source = df.load_from_pickle(str(ROOT / "examples/cylinder_flow_steady/model.pkl"))
results = []
for mu in (0.02, 0.2):
    for scale in (1.0, 10.0):
        model = ScaledPINN(pressure_scale=scale)
        model.load_state_dict(source.state_dict())
        model = model.to(df.device)
        label = f"transfer_mu{str(mu).replace('.', '')}_ps{scale:g}"
        domain = build_domain(mu, n_bound=300, n_interior=3000)
        metrics = evaluate(model, domain, mu, label)
        results.append(metrics)

(HERE / "transfer_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for item in results:
    print(
        item["label"],
        "relL2(u,p,speed)=",
        item["relative_l2_u"],
        item["relative_l2_p"],
        item["relative_l2_speed"],
        "losses=",
        item["losses"],
    )
