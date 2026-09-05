#!/usr/bin/env python3
"""Orchestrate the four-cell cavity PINN/QCPINN benchmark."""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark import available_variants  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Run the cavity PINN/QCPINN benchmark."
    )
    for flag, dest, help_text in (
        ("--pinn-uvp", "pinn_uvp", "Run PINN direct (u,v,p)."),
        ("--pinn-psip", "pinn_psip", "Run PINN streamfunction (psi,p)."),
        ("--qcpinn-uvp", "qcpinn_uvp", "Run QCPINN direct (u,v,p)."),
        ("--qcpinn-psip", "qcpinn_psip", "Run QCPINN streamfunction (psi,p)."),
    ):
        parser.add_argument(flag, dest=dest, action="store_true", help=help_text)
    parser.add_argument("--reference", action="store_true", help="Generate the FEM reference.")
    parser.add_argument("--compare", action="store_true", help="Generate comparison artifacts.")
    parser.add_argument("--all", action="store_true", help="Run reference, all cells, and comparison.")
    parser.add_argument("--smoke", action="store_true", help="Use the small smoke configuration.")
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--epochs_adam", type=int, default=None)
    parser.add_argument("--epochs_lbfgs", type=int, default=None)
    args = parser.parse_args()

    setup_flags = [args.pinn_uvp, args.pinn_psip, args.qcpinn_uvp, args.qcpinn_psip]
    if not any(setup_flags + [args.reference, args.compare, args.all]):
        args.all = True
    if args.all:
        args.reference = args.pinn_uvp = args.pinn_psip = True
        args.qcpinn_uvp = args.qcpinn_psip = args.compare = True
    if args.num_runs is not None and args.num_runs < 1:
        parser.error("--num_runs must be positive")
    if args.epochs_adam is not None and args.epochs_adam < 0:
        parser.error("--epochs_adam must be non-negative")
    if args.epochs_lbfgs is not None and args.epochs_lbfgs < 0:
        parser.error("--epochs_lbfgs must be non-negative")

    (SCRIPT_DIR / "results").mkdir(parents=True, exist_ok=True)
    smoke_variants = set(available_variants()) if args.smoke else set()

    def should_run(variant, selected):
        if not selected:
            return False
        if args.smoke and variant not in smoke_variants:
            print(f"Skipping {variant}: optional QCPINN backend is unavailable")
            return False
        return True

    def run(script_name, label, benchmark=False, compare=False):
        print("\n" + "=" * 72)
        print(f"Running {label} ...")
        print("=" * 72)
        command = [sys.executable, str(SCRIPT_DIR / script_name)]
        if args.smoke and benchmark:
            command.append("--smoke")
        if benchmark:
            for name, value in (
                ("--num_runs", args.num_runs),
                ("--epochs_adam", args.epochs_adam),
                ("--epochs_lbfgs", args.epochs_lbfgs),
            ):
                if value is not None:
                    command += [name, str(value)]
        if args.smoke and compare:
            command += ["--smoke", "--no-reference"]
        code = subprocess.call(command, cwd=str(SCRIPT_DIR))
        if code:
            raise SystemExit(f"{label} exited with code {code}")

    if args.reference and not args.smoke:
        run("reference.py", "fresh FEM reference")
    if should_run("PINN-UVP", args.pinn_uvp):
        run("benchmark_pinn_uvp.py", "PINN-UVP", benchmark=True)
    if should_run("PINN-PSIP", args.pinn_psip):
        run("benchmark_pinn_psip.py", "PINN-PSIP", benchmark=True)
    if should_run("QCPINN-UVP", args.qcpinn_uvp):
        run("benchmark_qcpinn_uvp.py", "QCPINN-UVP", benchmark=True)
    if should_run("QCPINN-PSIP", args.qcpinn_psip):
        run("benchmark_qcpinn_psip.py", "QCPINN-PSIP", benchmark=True)
    if args.compare:
        run("compare.py", "comparison", compare=True)
    print(f"\nAll requested stages complete. Results: {SCRIPT_DIR / 'results'}")


if __name__ == "__main__":
    main()
