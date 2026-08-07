#!/usr/bin/env python3
"""Orchestrate the direct versus stream-function cavity benchmark."""

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Run the direct (u,v,p) versus stream-function (psi,p) benchmark."
    )
    parser.add_argument("--uvp", action="store_true", help="Run the direct PINN.")
    parser.add_argument("--psip", action="store_true", help="Run the stream-function PINN.")
    parser.add_argument("--compare", action="store_true", help="Generate comparison plots and report.")
    parser.add_argument("--all", action="store_true", help="Run both models and compare them (default).")
    parser.add_argument("--num_runs", type=int, default=None, help="Override the number of runs.")
    parser.add_argument("--epochs_adam", type=int, default=None, help="Override Adam epochs.")
    parser.add_argument("--epochs_lbfgs", type=int, default=None, help="Override L-BFGS epochs.")
    args = parser.parse_args()

    if not any([args.uvp, args.psip, args.compare, args.all]):
        args.all = True
    if args.all:
        args.uvp = True
        args.psip = True
        args.compare = True

    results_dir = _SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def _run(script_name, label, benchmark_script=False):
        print("\n" + "=" * 60)
        print(f"Running {label} ...")
        print("=" * 60)
        cmd = [sys.executable, str(_SCRIPT_DIR / script_name)]
        if benchmark_script:
            if args.num_runs is not None:
                cmd += ["--num_runs", str(args.num_runs)]
            if args.epochs_adam is not None:
                cmd += ["--epochs_adam", str(args.epochs_adam)]
            if args.epochs_lbfgs is not None:
                cmd += ["--epochs_lbfgs", str(args.epochs_lbfgs)]
        rc = subprocess.call(cmd, cwd=str(_SCRIPT_DIR))
        if rc != 0:
            print(f"[ERROR] {label} exited with code {rc}")
            sys.exit(rc)

    if args.uvp:
        _run("benchmark_uvp.py", "direct (u,v,p) PINN", benchmark_script=True)
    if args.psip:
        _run("benchmark_psip.py", "stream-function (psi,p) PINN", benchmark_script=True)
    if args.compare:
        _run("compare.py", "formulation comparison")

    print("\n" + "=" * 60)
    print("All requested benchmarks complete.")
    print(f"Results: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
