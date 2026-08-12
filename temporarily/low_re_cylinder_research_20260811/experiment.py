"""Controlled DeepFlow cylinder-flow experiments; all outputs stay in this folder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import deepflow as df  # noqa: E402


OUT = Path(__file__).resolve().parent
HEIGHT = 0.41


class ScaledPINN(df.PINN):
    def __init__(self, pressure_scale=1.0, normalize_inputs=False, hard_pressure_outlet=False):
        self.pressure_scale = float(pressure_scale)
        self.normalize_inputs = bool(normalize_inputs)
        self.hard_pressure_outlet = bool(hard_pressure_outlet)
        super().__init__(
            width=32,
            length=5,
            input_vars=["x", "y"],
            output_vars=["u", "v", "p"],
        )

    def forward(self, inputs_dict):
        model_inputs = inputs_dict
        if self.normalize_inputs:
            model_inputs = {
                **inputs_dict,
                "x": 2.0 * inputs_dict["x"] / 1.1 - 1.0,
                "y": 2.0 * inputs_dict["y"] / HEIGHT - 1.0,
            }
        result = super().forward(model_inputs)
        pressure_factor = 1.1 - inputs_dict["x"] if self.hard_pressure_outlet else 1.0
        result["p"] = self.pressure_scale * pressure_factor * result["p"]
        return result


class NormalizedNavierStokes(df.NavierStokes):
    """Same PDE zero set, with residual equations multiplied by constants."""

    def __init__(self, *args, continuity_scale=1.0, momentum_scale=1.0, **kwargs):
        self.continuity_scale = float(continuity_scale)
        self.momentum_scale = float(momentum_scale)
        super().__init__(*args, **kwargs)

    def compute_residuals(self, inputs_dict):
        continuity, x_momentum, y_momentum = super().compute_residuals(inputs_dict)
        continuity = self.continuity_scale * continuity
        x_momentum = self.momentum_scale * x_momentum
        y_momentum = self.momentum_scale * y_momentum
        self.residual_fields = (continuity, x_momentum, y_momentum)
        self.var["continuity_residual"] = continuity
        self.var["x_momentum_residual"] = x_momentum
        self.var["y_momentum_residual"] = y_momentum
        return self.residual_fields


def build_domain(
    mu,
    n_bound=300,
    n_interior=3000,
    outlet_gradients=False,
    continuity_scale=1.0,
    momentum_scale=1.0,
):
    circle = df.geometry.circle(0.2, 0.2, 0.05)
    rectangle = df.geometry.rectangle([0.0, 1.1], [0.0, HEIGHT])
    area = rectangle - circle
    domain = df.domain(area, circle.bound_list)
    domain.bound_list[0].define_bc(
        {"u": ["y", lambda y: 4.0 * (HEIGHT - y) * y / HEIGHT**2], "v": 0}
    )
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    outlet = {"p": 0}
    if outlet_gradients:
        outlet.update({"u_x": 0, "v_x": 0})
    domain.bound_list[2].define_bc(outlet)
    for index in (3, 4, 5):
        domain.bound_list[index].define_bc({"u": 0, "v": 0})
    area.define_pde(
        NormalizedNavierStokes(
            U=1.0,
            L=1.0,
            mu=mu,
            rho=1.0,
            continuity_scale=continuity_scale,
            momentum_scale=momentum_scale,
        )
    )
    if n_bound and n_interior:
        domain.sampling_lhs([n_bound] * 6, [n_interior])
    return domain


def make_reference(mu, eval_grid=(121, 61)):
    df.manual_seed(69)
    domain = build_domain(mu, n_bound=0, n_interior=0)
    start = time.perf_counter()
    reference = domain.solve_fem(
        mesh_size=0.025,
        boundary_resolution=192,
        tolerance=1e-8,
        max_iterations=300,
        area_sampling_res=list(eval_grid),
        bound_sampling_res=241,
    )
    runtime = time.perf_counter() - start
    data = reference.area_evaluators[0].data_dict
    metadata = dict(reference.metadata)
    path = OUT / f"fem_mu_{mu:g}.npz"
    np.savez(
        path,
        **{k.replace("_ref", ""): np.asarray(v) for k, v in data.items() if k in {"x", "y", "u_ref", "v_ref", "p_ref"}},
        runtime_s=runtime,
        metadata_json=json.dumps(metadata, default=str),
    )
    summary = {
        "mu": mu,
        "runtime_s": runtime,
        "metadata": metadata,
        "ranges": {
            field: [float(np.min(data[field + "_ref"])), float(np.max(data[field + "_ref"]))]
            for field in ("u", "v", "p")
        },
    }
    (OUT / f"fem_mu_{mu:g}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def relative_l2(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def evaluate(model, domain, mu, label, eval_grid=(121, 61)):
    reference = np.load(OUT / f"fem_mu_{mu:g}.npz")
    area = domain.area_list[0]
    area.sampling_area(list(eval_grid))
    data = area.evaluate(model).data_dict
    if not (np.allclose(data["x"], reference["x"]) and np.allclose(data["y"], reference["y"])):
        raise RuntimeError("PINN and FEM grids do not align")
    u, v, p = (np.asarray(data[k]) for k in ("u", "v", "p"))
    ur, vr, pr = (np.asarray(reference[k]) for k in ("u", "v", "p"))
    speed = np.hypot(u, v)
    speed_ref = np.hypot(ur, vr)
    losses = domain._batched_loss(model)
    metrics = {
        "label": label,
        "mu": mu,
        "deepflow_Re": 1.0 / mu,
        "cylinder_Re_D": 0.1 / mu,
        "relative_l2_u": relative_l2(u, ur),
        "relative_l2_v": relative_l2(v, vr),
        "relative_l2_p": relative_l2(p, pr),
        "relative_l2_speed": relative_l2(speed, speed_ref),
        "mae_u": float(np.mean(np.abs(u - ur))),
        "mae_p": float(np.mean(np.abs(p - pr))),
        "pred_ranges": {k: [float(np.min(data[k])), float(np.max(data[k]))] for k in ("u", "v", "p")},
        "ref_ranges": {k: [float(np.min(reference[k])), float(np.max(reference[k]))] for k in ("u", "v", "p")},
        "mean_abs_residuals": {
            k: float(np.mean(np.abs(data[k + "_residual"])))
            for k in ("continuity", "x_momentum", "y_momentum")
        },
        "losses": {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in losses.items()},
    }
    np.savez(OUT / f"fields_{label}.npz", **{k: np.asarray(v) for k, v in data.items() if isinstance(v, (np.ndarray, list))})
    (OUT / f"metrics_{label}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train(args):
    df.manual_seed(args.seed)
    domain = build_domain(
        args.mu,
        n_bound=args.n_bound,
        n_interior=args.n_interior,
        outlet_gradients=args.outlet_gradients,
        continuity_scale=args.continuity_scale,
        momentum_scale=args.momentum_scale,
    )
    model = ScaledPINN(args.pressure_scale, args.normalize_inputs, args.hard_pressure_outlet)
    adam_loss = df.calc_loss_weighted(
        domain,
        bc_weights=args.adam_bc_weight,
        pde_weights=args.adam_pde_weight,
    )
    lbfgs_loss = df.calc_loss_weighted(
        domain,
        bc_weights=1.0,
        pde_weights=args.lbfgs_pde_weight,
    )
    start = time.perf_counter()
    _, best = model.train_adam(
        learning_rate=args.learning_rate,
        epochs=args.adam_epochs,
        calc_loss=adam_loss,
        print_every=max(1, args.adam_epochs // 10),
    )
    adam_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, best = best.train_lbfgs(
        epochs=args.lbfgs_epochs,
        calc_loss=lbfgs_loss,
        print_every=max(1, args.lbfgs_epochs // 10),
    )
    lbfgs_seconds = time.perf_counter() - start
    metrics = evaluate(best, domain, args.mu, args.label)
    metrics.update(
        {
            "pressure_scale": args.pressure_scale,
            "normalize_inputs": args.normalize_inputs,
            "hard_pressure_outlet": args.hard_pressure_outlet,
            "outlet_gradients": args.outlet_gradients,
            "continuity_scale": args.continuity_scale,
            "momentum_scale": args.momentum_scale,
            "adam_bc_weight": args.adam_bc_weight,
            "adam_pde_weight": args.adam_pde_weight,
            "lbfgs_pde_weight": args.lbfgs_pde_weight,
            "adam_seconds": adam_seconds,
            "lbfgs_seconds": lbfgs_seconds,
            "adam_epochs": args.adam_epochs,
            "lbfgs_epochs": args.lbfgs_epochs,
            "n_bound": args.n_bound,
            "n_interior": args.n_interior,
            "seed": args.seed,
        }
    )
    (OUT / f"metrics_{args.label}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"state_dict": best.state_dict(), "metrics": metrics}, OUT / f"model_{args.label}.pt")
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--pressure-scale", type=float, default=1.0)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--hard-pressure-outlet", action="store_true")
    parser.add_argument("--outlet-gradients", action="store_true")
    parser.add_argument("--continuity-scale", type=float, default=1.0)
    parser.add_argument("--momentum-scale", type=float, default=1.0)
    parser.add_argument("--adam-bc-weight", type=float, default=1.0)
    parser.add_argument("--adam-pde-weight", type=float, default=1.0)
    parser.add_argument("--lbfgs-pde-weight", type=float, default=1.0)
    parser.add_argument("--adam-epochs", type=int, default=600)
    parser.add_argument("--lbfgs-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--n-bound", type=int, default=300)
    parser.add_argument("--n-interior", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=69)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.reference:
        make_reference(arguments.mu)
    else:
        train(arguments)
