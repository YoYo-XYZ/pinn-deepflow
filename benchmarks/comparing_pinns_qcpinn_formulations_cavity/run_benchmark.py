#!/usr/bin/env python3
"""Orchestrate the four-cell PINN/QCPINN cavity benchmark."""

import argparse
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run PINN and QCPINN cavity benchmarks with UVP and PSIP "
            "Navier-Stokes formulations."
        )
    )
    parser.add_argument(
        "--pinn-uvp", "--pinn_uvp", dest="pinn_uvp", action="store_true",
        help="Run the PINN direct (u,v,p) setup.",
    )
    parser.add_argument(
        "--pinn-psip", "--pinn_psip", dest="pinn_psip", action="store_true",
        help="Run the PINN stream-function (psi,p) setup.",
    )
    parser.add_argument(
        "--qcpinn-uvp", "--qcpinn_uvp", dest="qcpinn_uvp", action="store_true",
        help="Run the QCPINN direct (u,v,p) setup.",
    )
    parser.add_argument(
        "--qcpinn-psip", "--qcpinn_psip", dest="qcpinn_psip", action="store_true",
        help="Run the QCPINN stream-function (psi,p) setup.",
    )
    parser.add_argument(
        "--reference", action="store_true",
        help="Generate fresh 101x101 and 201x201 CFD references.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate comparison plots and the Markdown report.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run the CFD reference, all four setups, and comparison (default).",
    )
    parser.add_argument(
        "--num_runs", type=int, default=None,
        help="Override the default number of runs for selected setups.",
    )
    parser.add_argument(
        "--epochs_adam", type=int, default=None,
        help="Override Adam epochs for selected setups.",
    )
    parser.add_argument(
        "--epochs_lbfgs", type=int, default=None,
        help="Override L-BFGS epochs for selected setups.",
    )
    args = parser.parse_args()

    setup_flags = [args.pinn_uvp, args.pinn_psip, args.qcpinn_uvp, args.qcpinn_psip]
    if not any(setup_flags + [args.reference, args.compare, args.all]):
        args.all = True
    if args.all:
        args.pinn_uvp = True
        args.pinn_psip = True
        args.qcpinn_uvp = True
        args.qcpinn_psip = True
        args.reference = True
        args.compare = True

    if args.num_runs is not None and args.num_runs < 1:
        parser.error("--num_runs must be at least 1")
    if args.epochs_adam is not None and args.epochs_adam < 0:
        parser.error("--epochs_adam must be non-negative")
    if args.epochs_lbfgs is not None and args.epochs_lbfgs < 0:
        parser.error("--epochs_lbfgs must be non-negative")

    results_dir = _SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def _run(script_name, label, benchmark=False):
        print("\n" + "=" * 68)
        print(f"Running {label} ...")
        print("=" * 68)
        command = [sys.executable, str(_SCRIPT_DIR / script_name)]
        if benchmark:
            if args.num_runs is not None:
                command += ["--num_runs", str(args.num_runs)]
            if args.epochs_adam is not None:
                command += ["--epochs_adam", str(args.epochs_adam)]
            if args.epochs_lbfgs is not None:
                command += ["--epochs_lbfgs", str(args.epochs_lbfgs)]
        return_code = subprocess.call(command, cwd=str(_SCRIPT_DIR))
        if return_code != 0:
            print(f"[ERROR] {label} exited with code {return_code}")
            raise SystemExit(return_code)

    if args.reference:
        _run("reference.py", "fresh finite-volume CFD reference")
    if args.pinn_uvp:
        _run("benchmark_pinn_uvp.py", "PINN-UVP benchmark", benchmark=True)
    if args.pinn_psip:
        _run("benchmark_pinn_psip.py", "PINN-PSIP benchmark", benchmark=True)
    if args.qcpinn_uvp:
        _run("benchmark_qcpinn_uvp.py", "QCPINN-UVP benchmark", benchmark=True)
    if args.qcpinn_psip:
        _run("benchmark_qcpinn_psip.py", "QCPINN-PSIP benchmark", benchmark=True)
    if args.compare:
        _run("compare.py", "four-cell comparison")

    print("\n" + "=" * 68)
    print("All requested benchmark stages complete.")
    print(f"Results: {results_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
