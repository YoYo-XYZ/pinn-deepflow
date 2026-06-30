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
import os
import subprocess
import sys


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
        default=True,
        help="Run all benchmarks and comparison (default)",
    )
    args = parser.parse_args()

    # If no specific flags are given, default to --all
    if not any([args.deepflow, args.deepxde, args.compare]):
        args.all = True

    if args.all:
        args.deepflow = True
        args.deepxde = True
        args.compare = True

    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    base = os.path.dirname(os.path.abspath(__file__))

    if args.deepflow:
        print("\n" + "=" * 60)
        print("Running DeepFlow benchmark ...")
        print("=" * 60)
        rc = subprocess.call(
            [sys.executable, os.path.join(base, "benchmark_deepflow.py")]
        )
        if rc != 0:
            print(f"[ERROR] DeepFlow benchmark exited with code {rc}")
            sys.exit(rc)

    if args.deepxde:
        print("\n" + "=" * 60)
        print("Running DeepXDE benchmark ...")
        print("=" * 60)
        rc = subprocess.call(
            [sys.executable, os.path.join(base, "benchmark_deepxde.py")]
        )
        if rc != 0:
            print(f"[ERROR] DeepXDE benchmark exited with code {rc}")
            sys.exit(rc)

    if args.compare:
        print("\n" + "=" * 60)
        print("Running comparison ...")
        print("=" * 60)
        rc = subprocess.call(
            [sys.executable, os.path.join(base, "compare.py")]
        )
        if rc != 0:
            print(f"[ERROR] Comparison script exited with code {rc}")
            sys.exit(rc)

    print("\nBenchmark suite complete.")


if __name__ == "__main__":
    main()
