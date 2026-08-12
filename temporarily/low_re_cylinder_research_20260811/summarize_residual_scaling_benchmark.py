"""Aggregate the matched three-seed residual-scaling benchmark."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

groups = {
    "momentum_div30": [
        "conditioned_mu02",
        "bench_momentum_div30_s70",
        "bench_momentum_div30_s71",
    ],
    "continuity_mul30": [
        "bench_continuity_mul30_s69",
        "bench_continuity_mul30_s70",
        "bench_continuity_mul30_s71",
    ],
}

summary = {}
for group, labels in groups.items():
    runs = [json.loads((HERE / f"metrics_{label}.json").read_text()) for label in labels]
    summary[group] = {"runs": labels}
    for metric in ("relative_l2_u", "relative_l2_v", "relative_l2_speed", "relative_l2_p"):
        values = np.array([run[metric] for run in runs])
        summary[group][metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
        }
    bc = np.array([run["losses"]["bc_loss"] for run in runs])
    summary[group]["bc_loss"] = {
        "values": bc.tolist(),
        "mean": float(bc.mean()),
        "sample_std": float(bc.std(ddof=1)),
    }

weighted = json.loads(
    (HERE / "metrics_bench_continuity_mul30_weighted_s69.json").read_text()
)
summary["continuity_mul30_with_pde_weight_1_over_900"] = {
    metric: weighted[metric]
    for metric in ("relative_l2_u", "relative_l2_v", "relative_l2_speed", "relative_l2_p")
}
summary["continuity_mul30_with_pde_weight_1_over_900"]["bc_loss"] = weighted["losses"]["bc_loss"]

(HERE / "residual_scaling_benchmark.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

labels = ["momentum / 30", "continuity x 30"]
metrics = ["relative_l2_speed", "relative_l2_p"]
colors = ["#2878b5", "#d9534f"]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
for ax, metric, title in zip(axes, metrics, ["Velocity-magnitude error", "Pressure error"]):
    means = [100 * summary[group][metric]["mean"] for group in groups]
    stds = [100 * summary[group][metric]["sample_std"] for group in groups]
    ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel("Relative L2 error (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
fig.suptitle("Raw residual scaling benchmark at mu=0.2 (mean +/- sample SD, 3 seeds)")
fig.savefig(HERE / "residual_scaling_benchmark.png", dpi=180)

print(json.dumps(summary, indent=2))
