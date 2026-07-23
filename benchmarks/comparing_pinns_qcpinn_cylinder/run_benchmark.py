#!/usr/bin/env python3
"""
Orchestrator for the QCPINN vs PINN cylinder flow benchmark.

Usage:
    python run_benchmark.py --all            # run everything (default)
    python run_benchmark.py --pinn           # classical PINN only
    python run_benchmark.py --qcpinn         # QCPINN only
    python run_benchmark.py --compare        # comparison only (uses existing NPZs)
    python run_benchmark.py --pinn --qcpinn --compare

    python run_benchmark.py --all --num_runs 1   # single run (smoke test)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Run the QCPINN vs PINN 2D cylinder flow benchmark."
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
        "--compare", action="store_true",
        help="Run the comparison script (compare.py).",
    )
    parser.add_argument(
        "--all", action="store_true", default=True,
        help="Run all benchmarks and comparison (default).",
    )
    parser.add_argument(
        "--num_runs", type=int, default=None,
        help="Override the default number of runs (forwarded to both benchmarks).",
    )
    args = parser.parse_args()

    # Default to --all if no specific flag was given
    if not any([args.pinn, args.qcpinn, args.compare]):
        args.all = True
    if args.all:
        args.pinn = True
        args.qcpinn = True
        args.compare = True

    # Ensure results directory exists
    results_dir = _SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    def _run(script_name, label):
        print("\n" + "=" * 60)
        print(f"Running {label} ...")
        print("=" * 60)
        cmd = [sys.executable, str(_SCRIPT_DIR / script_name)]
        if args.num_runs is not None:
            cmd += ["--num_runs", str(args.num_runs)]
        rc = subprocess.call(cmd, cwd=str(_SCRIPT_DIR))
        if rc != 0:
            print(f"[ERROR] {label} exited with code {rc}")
            sys.exit(rc)

    if args.pinn:
        _run("benchmark_pinn.py", "PINN benchmark")
    if args.qcpinn:
        _run("benchmark_qcpinn.py", "QCPINN benchmark")
    if args.compare:
        _run("compare.py", "comparison")

    print("\n" + "=" * 60)
    print("All requested benchmarks complete.")
    print(f"Results: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
