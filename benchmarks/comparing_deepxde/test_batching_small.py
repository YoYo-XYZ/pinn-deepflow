#!/usr/bin/env python3
"""Check equivalence between batched and per-geometry loss calculation."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402
import deepflow as df  # noqa: E402


def _value(value):
    return value.item() if isinstance(value, torch.Tensor) else value


def main():
    df.manual_seed(69)

    # Channel-flow setup, matching benchmark_deepflow.py.
    rect = df.geometry.rectangle([0, 5.0], [0, 1.0])
    domain = df.domain(rect)
    domain.bound_list[0].define_bc({"u": 1, "v": 0})
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.bound_list[3].define_bc({"u": 0, "v": 0})
    domain.area_list[0].define_pde(
        df.pde.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000)
    )
    domain.sampling_random([100, 500, 100, 500], [2000])

    model = df.PINN(
        width=32,
        length=4,
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
    ).to(df.get_device())

    old_loss = {"pde_loss": 0.0, "bc_loss": 0.0, "ic_loss": 0.0}
    for geometry in domain:
        key = f"{geometry.physics_type.lower()}_loss"
        old_loss[key] += geometry.calc_loss(model)
    old_loss["total_loss"] = sum(
        value for key, value in old_loss.items() if key != "total_loss"
    )
    new_loss = df.calc_loss_simple(domain)(model)

    print("Old (per-geometry):", {key: _value(value) for key, value in old_loss.items()})
    print("New (batched):     ", {key: _value(value) for key, value in new_loss.items()})
    print()

    all_ok = True
    for key, old_value in old_loss.items():
        old_value = _value(old_value)
        new_value = _value(new_loss[key])
        difference = abs(old_value - new_value)
        ok = difference < 1e-5
        print(
            f"  [{'OK  ' if ok else 'FAIL'}] {key:12s}: "
            f"old={old_value:.10f}  new={new_value:.10f} "
            f" diff={difference:.2e}"
        )
        all_ok &= ok

    print()
    if all_ok:
        print("All losses match — batched path is numerically equivalent.")
        return 0

    print("MISMATCH detected — batched path differs from per-geometry path.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
