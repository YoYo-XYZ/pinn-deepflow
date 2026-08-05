#!/usr/bin/env python3
"""Orchestrate the PINN/QCPINN lid-driven cavity benchmark."""

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Run the PINN/QCPINN benchmark with an optional CFD reference."
    )
    parser.add_argument(
        "--pinn", action="store_true",
        help="Run the classical PINN benchmark (benchmark_pinn.py).",
    )
    parser.add_argument(
        "--qcpinn", action="store_true",
        help="Run the QCPINN benchmark (benchmark_qcpinn.py).",
    )
    parser.add_argument(
        "--reference", action="store_true",
        help="Generate the 101x101 and 201x201 CFD reference solutions.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run the comparison script (compare.py).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all benchmarks and comparison (default).",
    )
    parser.add_argument(
        "--num_runs", type=int, default=None,
        help="Override the default number of runs for selected benchmarks.",
    )
    args = parser.parse_args()

    if not any([args.pinn, args.qcpinn, args.reference, args.compare, args.all]):
        args.all = True
    if args.all:
        args.pinn = True
        args.qcpinn = True
        args.reference = True
        args.compare = True

    results_dir = _SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def _run(script_name, label, extra_args=None):
        print("\n" + "=" * 60)
        print(f"Running {label} ...")
        print("=" * 60)
        cmd = [sys.executable, str(_SCRIPT_DIR / script_name)]
        if extra_args:
            cmd += extra_args
        if args.num_runs is not None and script_name in {
            "benchmark_pinn.py",
            "benchmark_qcpinn.py",
        }:
            cmd += ["--num_runs", str(args.num_runs)]
        rc = subprocess.call(cmd, cwd=str(_SCRIPT_DIR))
        if rc != 0:
            print(f"[ERROR] {label} exited with code {rc}")
            sys.exit(rc)

    if args.pinn:
        _run("benchmark_pinn.py", "PINN benchmark")
    if args.qcpinn:
        _run("benchmark_qcpinn.py", "QCPINN benchmark")
    if args.reference:
        _run(
            "reference_cfd.py",
            "finite-volume CFD reference",
            extra_args=["--grid_convergence"],
        )
    if args.compare:
        _run("compare.py", "comparison")

    print("\n" + "=" * 60)
    print("All requested benchmarks complete.")
    print(f"Results: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
