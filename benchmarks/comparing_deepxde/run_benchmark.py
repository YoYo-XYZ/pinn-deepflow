#!/usr/bin/env python3
"""
Orchestrator for the 2D steady channel flow benchmark.

Usage:
    python run_benchmark.py --all         # run everything (default)
    python run_benchmark.py --deepflow    # DeepFlow only
    python run_benchmark.py --deepxde     # DeepXDE only
    python run_benchmark.py --compare     # compare existing results only
    python run_benchmark.py --deepflow --deepxde --compare
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _run(label, script_name):
    print(f"\n{'=' * 60}")
    print(f"Running {label} ...")
    print("=" * 60)
    result = subprocess.run([sys.executable, str(SCRIPT_DIR / script_name)])
    if result.returncode:
        print(f"[ERROR] {label} exited with code {result.returncode}")
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the 2D channel flow benchmark (DeepFlow vs DeepXDE)."
    )
    parser.add_argument(
        "--deepflow", action="store_true", help="Run the DeepFlow benchmark"
    )
    parser.add_argument(
        "--deepxde", action="store_true", help="Run the DeepXDE benchmark"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run the comparison script"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks and comparison (default)",
    )
    args = parser.parse_args()

    run_all = args.all or not any((args.deepflow, args.deepxde, args.compare))
    if run_all or args.deepflow:
        _run("DeepFlow benchmark", "benchmark_deepflow.py")
    if run_all or args.deepxde:
        _run("DeepXDE benchmark", "benchmark_deepxde.py")
    if run_all or args.compare:
        _run("comparison", "compare.py")

    print("\nBenchmark suite complete.")


if __name__ == "__main__":
    main()
