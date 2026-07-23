"""
Minimal QCPINN smoke test: verify the full pipeline works (no L-BFGS).
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

import deepflow as df

# 1. Build domain (same as benchmark)
df.manual_seed(69)
circle = df.geometry.circle(0.2, 0.2, 0.05)
rectangle = df.geometry.rectangle([0, 1.1], [0, 0.41])
area = rectangle - circle
domain = df.domain(area, circle.bound_list)
domain.area_list[0].define_pde(df.pde.NavierStokes(U=1, L=1, mu=0.02, rho=1))
domain.bound_list[0].define_bc({"u": ["y", lambda y: 4 * y * (0.41 - y) / 0.41 ** 2], "v": 0})
domain.bound_list[1].define_bc({"u": 0, "v": 0})
domain.bound_list[2].define_bc({"p": 0})
domain.bound_list[3].define_bc({"u": 0, "v": 0})
domain.bound_list[4].define_bc({"u": 0, "v": 0})
domain.bound_list[5].define_bc({"u": 0, "v": 0})
# Tiny sampling for speed
domain.sampling_lhs([100, 100, 100, 100, 100, 100], [400])

# 2. Build QCPINN
model0 = df.QCPINN(
    input_vars=["x", "y"],
    output_vars=["u", "v", "p"],
    hidden_layer_pre=[50],
    hidden_layer_post=[50],
    nqubits=4,
    q_layer_iterations=1,
)
n_params = sum(p.numel() for p in model0.parameters())
print(f"QCPINN params: {n_params} (expected 769)")

# 3. Train Adam only (20 epochs, no L-BFGS)
calc_loss = df.calc_loss_simple(domain)
m, mb = model0.train_adam(
    calc_loss=calc_loss,
    learning_rate=0.004,
    epochs=20,
    threshold_loss=None,
    print_every=5,
)
final = m.loss_history["total_loss"][-1]
print(f"Adam done. final loss = {final:.4e}")

# 4. Quick eval
ae = domain.area_list[0].evaluate(mb)
ae.sampling_area([50, 25])
print(f"Eval data keys (first 6): {sorted(ae.data_dict.keys())[:6]}")
print(f"u field shape: {ae.data_dict['u'].shape}")
print("QCPINN pipeline OK.")
