#!/usr/bin/env python3
"""
Reproducibility verification script for deepflow.

Tests:
  1. Micro LHS reproducibility (same seed -> identical samples)
  2. Two full CPU training runs (seed=69, deterministic=False) -- bit-exact equality
  3. deterministic=True on CPU matches default run (seed=69)
  4. Different seed (42) produces different predictions (negative control)
  5. CUDA deterministic vs CPU comparison (if CUDA available)
"""

import copy
import math
import os
import sys
import traceback

import numpy as np
import torch

import deepflow as df

from torch import sin, pi

# -- Training hyperparameters (fast but meaningful) ----------------------
N_ADAM = 500
N_LBFGS = 100
N_IC = 2000
N_BC = 1000
N_PDE = 4000

# -- Test counters ------------------------------------------------------
PASS = 0
FAIL = 0
ERRORS = []


def report(msg: str, ok: bool = True):
    global PASS, FAIL
    if ok:
        print(f"  [OK] {msg}")
        PASS += 1
    else:
        print(f"  [FAIL] {msg}")
        FAIL += 1


# -- Helpers -------------------------------------------------------------

def tensors_equal(a, b):
    if isinstance(a, list):
        return all(torch.equal(x, y) for x, y in zip(a, b))
    return torch.equal(a, b)


def samples_equal(s1, s2):
    for k in s1:
        x1, y1 = s1[k]
        x2, y2 = s2[k]
        if not (torch.equal(x1, x2) and torch.equal(y1, y2)):
            return False
    return True


def loss_dicts_equal(h1, h2):
    for key in h1:
        if key not in h2:
            return False
        if len(h1[key]) != len(h2[key]):
            return False
        for v1, v2 in zip(h1[key], h2[key]):
            if v1 != v2:
                return False
    return True


# -- Problem setup -------------------------------------------------------

def create_burgers(seed=69, deterministic=False, device='cpu'):
    """Create domain and model for Burgers' equation, seeded."""
    # NOTE: df.device = device is NOT enough; get_device() reads
    # deepflow.utility.device directly. Set it at the source.
    import deepflow.utility as _df_util
    _df_util.device = device
    df.manual_seed(seed, deterministic=deterministic)

    area = df.geometry.rectangle([-1, 1], [0, 1])
    line_ic = df.geometry.line_horizontal(y=0, range_x=[-1, 1])
    line_bc1 = df.geometry.line_vertical(x=-1, range_y=[0, 1])
    line_bc2 = df.geometry.line_vertical(x=1, range_y=[0, 1])
    domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)

    # Physics
    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=0.01 / pi))
    domain.bound_list[0].define_bc({'u': ['x', lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({'u': 0})
    domain.bound_list[2].define_bc({'u': 0})

    model = df.PINN(input_vars=['x', 'y'], output_vars=['u'], width=16, length=4)
    return domain, model


def capture_samples(domain):
    out = {}
    for i, g in enumerate(domain.bound_list):
        out[f'bound_{i}'] = (g.X.clone(), g.Y.clone())
    for i, g in enumerate(domain.area_list):
        out[f'area_{i}'] = (g.X.clone(), g.Y.clone())
    return out


def train_burgers(domain, model):
    """Run full training (Adam -> LBFGS) and return diagnostic dict."""
    calc_loss = df.calc_loss_weighted(domain, bc_weights=1)

    def do_between(epoch, m):
        if epoch % 500 == 0:
            domain.sampling_R3([N_IC, N_BC, N_BC], [N_PDE])

    # Initial sampling
    domain.sampling_lhs([N_IC, N_BC, N_BC], [N_PDE])

    initial_samples = capture_samples(domain)
    initial_weights = [p.clone() for p in model.parameters()]

    # Adam
    model, _ = model.train_adam(
        calc_loss=calc_loss,
        learning_rate=0.004,
        epochs=N_ADAM,
        do_between_epochs=do_between,
        print_every=N_ADAM,
    )
    loss_adam = copy.deepcopy(model.loss_history)

    # LBFGS
    model, _ = model.train_lbfgs(
        calc_loss=calc_loss,
        epochs=N_LBFGS,
        print_every=N_LBFGS,
    )

    final_weights = [p.clone() for p in model.parameters()]
    loss_full = copy.deepcopy(model.loss_history)

    return {
        'model': model,
        'initial_samples': initial_samples,
        'initial_weights': initial_weights,
        'loss_history_adam': loss_adam,
        'loss_history': loss_full,
        'final_weights': final_weights,
    }


def eval_on_grid(model):
    """Evaluate model on a fixed 101x101 grid, return 1-D CPU prediction tensor."""
    # Infer model device from first parameter
    model_dev = next(model.parameters()).device
    x = torch.linspace(-1, 1, 101)
    y = torch.linspace(0, 1, 101)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    inp = {'x': X.reshape(-1).to(model_dev), 'y': Y.reshape(-1).to(model_dev)}
    model.eval()
    with torch.no_grad():
        pred = model(inp)['u']
    return pred.cpu()


# =========================================================================
# TEST 1 -- Micro LHS reproducibility
# =========================================================================
print("\n" + "=" * 70)
print("TEST 1: Micro LHS reproducibility")
print("=" * 70)

df.manual_seed(42)
lhs_a = df.latin_hypercube_sampling(100, 2, [0.0, 0.0], [1.0, 1.0])

df.manual_seed(42)
lhs_b = df.latin_hypercube_sampling(100, 2, [0.0, 0.0], [1.0, 1.0])

report("Same seed -> identical LHS samples", torch.equal(lhs_a, lhs_b))

df.manual_seed(99)
lhs_c = df.latin_hypercube_sampling(100, 2, [0.0, 0.0], [1.0, 1.0])

report("Different seed -> different LHS samples", not torch.equal(lhs_a, lhs_c))

# =========================================================================
# TEST 2 -- Bit-exact CPU reproducibility (seed=69, deterministic=False)
# =========================================================================
print("\n" + "=" * 70)
print("TEST 2: Two identical CPU runs (seed=69, deterministic=False)")
print("=" * 70)

try:
    d1, m1 = create_burgers(69, False, 'cpu')
    r1 = train_burgers(d1, m1)
    report("Run 1 completed", True)
except Exception as e:
    report(f"Run 1 failed: {e}", False)
    ERRORS.append(traceback.format_exc())

try:
    d2, m2 = create_burgers(69, False, 'cpu')
    r2 = train_burgers(d2, m2)
    report("Run 2 completed", True)
except Exception as e:
    report(f"Run 2 failed: {e}", False)
    ERRORS.append(traceback.format_exc())

if 'r1' in locals() and 'r2' in locals():
    t2 = {}
    t2['init_samp'] = samples_equal(r1['initial_samples'], r2['initial_samples'])
    report("Initial samples match", t2['init_samp'])

    t2['init_w'] = tensors_equal(r1['initial_weights'], r2['initial_weights'])
    report("Initial weights match", t2['init_w'])

    t2['loss_adam'] = loss_dicts_equal(r1['loss_history_adam'], r2['loss_history_adam'])
    report("Adam loss history matches", t2['loss_adam'])

    t2['loss_full'] = loss_dicts_equal(r1['loss_history'], r2['loss_history'])
    report("Full loss history matches", t2['loss_full'])

    t2['final_w'] = tensors_equal(r1['final_weights'], r2['final_weights'])
    report("Final weights match", t2['final_w'])

    pred1 = eval_on_grid(r1['model'])
    pred2 = eval_on_grid(r2['model'])
    t2['pred'] = torch.equal(pred1, pred2)
    report("Final grid predictions match", t2['pred'])

    test2_pass = all(t2.values())
    if test2_pass:
        print("  >>> TEST 2 PASSED: Bit-exact CPU reproducibility confirmed.")
    else:
        print("  >>> TEST 2 FAILED")
else:
    test2_pass = False
    print("  >>> TEST 2 ABORTED (one or both runs failed)")

# =========================================================================
# TEST 3 -- deterministic=True on CPU (seed=69)
# =========================================================================
print("\n" + "=" * 70)
print("TEST 3: deterministic=True on CPU (seed=69)")
print("=" * 70)

try:
    d3, m3 = create_burgers(69, deterministic=True, device='cpu')
    r3 = train_burgers(d3, m3)
    report("Run 3 completed", True)

    if test2_pass:
        t3 = {}
        t3['init_samp'] = samples_equal(r1['initial_samples'], r3['initial_samples'])
        report("Initial samples match (vs default)", t3['init_samp'])

        t3['init_w'] = tensors_equal(r1['initial_weights'], r3['initial_weights'])
        report("Initial weights match (vs default)", t3['init_w'])

        t3['loss_full'] = loss_dicts_equal(r1['loss_history'], r3['loss_history'])
        report("Full loss history matches (vs default)", t3['loss_full'])

        t3['final_w'] = tensors_equal(r1['final_weights'], r3['final_weights'])
        report("Final weights match (vs default)", t3['final_w'])

        pred3 = eval_on_grid(r3['model'])
        t3['pred'] = torch.equal(pred1, pred3)
        report("Final grid predictions match (vs default)", t3['pred'])

        test3_pass = all(t3.values())
    else:
        # Can't compare if test2 failed; just record what we can
        test3_pass = False
        report("Skipping comparison (test 2 baseline missing)", False)

    if test3_pass:
        print("  >>> TEST 3 PASSED: deterministic=True matches default on CPU.")
    else:
        print("  >>> TEST 3 FAILED")

except Exception as e:
    report(f"Run 3 failed: {e}", False)
    ERRORS.append(traceback.format_exc())
    test3_pass = False

# =========================================================================
# TEST 4 -- Different seed (42) -> negative control
# =========================================================================
print("\n" + "=" * 70)
print("TEST 4: Different seed (42) -- negative control")
print("=" * 70)

try:
    d4, m4 = create_burgers(42, False, 'cpu')
    r4 = train_burgers(d4, m4)
    report("Run 4 completed", True)

    pred4 = eval_on_grid(r4['model'])
    ok_diff = not torch.equal(pred1, pred4)
    report("Final prediction differs from seed=69 run", ok_diff)

    if ok_diff:
        print("  >>> TEST 4 PASSED: Different seed produces different prediction.")
    else:
        print("  >>> TEST 4 FAILED: Different seed produced identical prediction.")
    test4_pass = ok_diff
except Exception as e:
    report(f"Run 4 failed: {e}", False)
    ERRORS.append(traceback.format_exc())
    test4_pass = False

# =========================================================================
# TEST 5 -- CUDA comparison (if available)
# =========================================================================
if torch.cuda.is_available():
    print("\n" + "=" * 70)
    print("TEST 5: CUDA deterministic vs CPU (seed=69)")
    print("=" * 70)

    try:
        d5, m5 = create_burgers(69, deterministic=True, device='cuda')
        r5 = train_burgers(d5, m5)
        report("CUDA run completed", True)

        pred5 = eval_on_grid(r5['model'])

        # Compare with CPU (pred1) -- move CPU reference to CUDA for diff
        max_diff = (pred5 - pred1.to(pred5.device)).abs().max().item()
        print(f"  CUDA vs CPU max absolute difference: {max_diff:.6e}")

        if max_diff < 1e-14:
            report("CUDA matches CPU within 1e-14", True)
        else:
            print(f"  Note: CUDA diff = {max_diff:.4e} (expected due to cuDNN non-determinism)")
            report("CUDA diff reported (not required to be zero)", True)

    except Exception as e:
        report(f"CUDA run failed: {e}", False)
        ERRORS.append(traceback.format_exc())
else:
    print("\n" + "=" * 70)
    print("TEST 5: CUDA not available -- skipping")
    print("=" * 70)

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 70)

if ERRORS:
    print("\nError tracebacks:")
    for i, tb in enumerate(ERRORS, 1):
        print(f"--- Exception {i} ---")
        print(tb)

sys.exit(0 if FAIL == 0 else 1)
