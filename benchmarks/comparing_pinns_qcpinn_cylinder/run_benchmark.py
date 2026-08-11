#!/usr/bin/env python3
"""Orchestrate the four-cell cylinder benchmark."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run the cylinder PINN/QCPINN benchmark.")
    for flag, dest, help_text in (
        ("--pinn-uvp", "pinn_uvp", "Run PINN direct (u,v,p)."),
        ("--pinn-psip", "pinn_psip", "Run PINN stream-function (psi,p)."),
        ("--qcpinn-uvp", "qcpinn_uvp", "Run QCPINN direct (u,v,p)."),
        ("--qcpinn-psip", "qcpinn_psip", "Run QCPINN stream-function (psi,p)."),
    ):
        parser.add_argument(flag, dest=dest, action="store_true", help=help_text)
    parser.add_argument("--reference", action="store_true", help="Generate the FEM reference.")
    parser.add_argument("--compare", action="store_true", help="Generate comparison artifacts.")
    parser.add_argument("--all", action="store_true", help="Run reference, all cells, and comparison.")
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--epochs_adam", type=int, default=None)
    parser.add_argument("--epochs_lbfgs", type=int, default=None)
    args = parser.parse_args()

    setup_flags = [args.pinn_uvp, args.pinn_psip, args.qcpinn_uvp, args.qcpinn_psip]
    if not any(setup_flags + [args.reference, args.compare, args.all]):
        args.all = True
    if args.all:
        args.reference = args.pinn_uvp = args.pinn_psip = args.qcpinn_uvp = args.qcpinn_psip = args.compare = True
    if args.num_runs is not None and args.num_runs < 1:
        parser.error("--num_runs must be positive")
    if args.epochs_adam is not None and args.epochs_adam < 0:
        parser.error("--epochs_adam must be non-negative")
    if args.epochs_lbfgs is not None and args.epochs_lbfgs < 0:
        parser.error("--epochs_lbfgs must be non-negative")

    (SCRIPT_DIR / "results").mkdir(parents=True, exist_ok=True)

    def run(script_name, label, benchmark=False):
        print("\n" + "=" * 72)
        print(f"Running {label} ...")
        print("=" * 72)
        command = [sys.executable, str(SCRIPT_DIR / script_name)]
        if benchmark:
            for name, value in (("--num_runs", args.num_runs), ("--epochs_adam", args.epochs_adam), ("--epochs_lbfgs", args.epochs_lbfgs)):
                if value is not None:
                    command += [name, str(value)]
        code = subprocess.call(command, cwd=str(SCRIPT_DIR))
        if code:
            raise SystemExit(f"{label} exited with code {code}")

    if args.reference:
        run("reference.py", "fresh FEM reference")
    if args.pinn_uvp:
        run("benchmark_pinn_uvp.py", "PINN-UVP", benchmark=True)
    if args.pinn_psip:
        run("benchmark_pinn_psip.py", "PINN-PSIP", benchmark=True)
    if args.qcpinn_uvp:
        run("benchmark_qcpinn_uvp.py", "QCPINN-UVP", benchmark=True)
    if args.qcpinn_psip:
        run("benchmark_qcpinn_psip.py", "QCPINN-PSIP", benchmark=True)
    if args.compare:
        run("compare.py", "comparison")
    print(f"\nAll requested stages complete. Results: {SCRIPT_DIR / 'results'}")


if __name__ == "__main__":
    main()
