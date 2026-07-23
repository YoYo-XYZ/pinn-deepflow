"""
Minimal QCPINN smoke test: verify the full pipeline works (no L-BFGS).
"""
from benchmark_common import build_domain, count_params, df
from common_config import QC_ITERATIONS, QC_NQUBITS, QC_POST, QC_PRE

df.manual_seed(69)
domain = build_domain([100] * 6, [400])

model0 = df.QCPINN(
    input_vars=["x", "y"],
    output_vars=["u", "v", "p"],
    hidden_layer_pre=QC_PRE,
    hidden_layer_post=QC_POST,
    nqubits=QC_NQUBITS,
    q_layer_iterations=QC_ITERATIONS,
)
n_params = count_params(model0)
assert n_params == 769, f"Expected 769 QCPINN parameters, got {n_params}"
print(f"QCPINN params: {n_params} (expected 769)")

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

ae = domain.area_list[0].evaluate(mb)
ae.sampling_area([50, 25])
print(f"Eval data keys (first 6): {sorted(ae.data_dict.keys())[:6]}")
print(f"u field shape: {ae.data_dict['u'].shape}")
print("QCPINN pipeline OK.")
