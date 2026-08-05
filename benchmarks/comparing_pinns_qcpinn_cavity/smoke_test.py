"""Minimal QCPINN smoke test for the lid-driven cavity pipeline."""

from benchmark_common import build_domain, count_params, df
from common_config import QC_ITERATIONS, QC_NQUBITS, QC_POST, QC_PRE


df.manual_seed(69)
domain = build_domain([100, 100, 100, 100, 2], [400])

assert len(domain.bound_list) == 5
assert domain.bound_list[0].condition_dict == {"u": 0, "v": 0}
assert domain.bound_list[1].condition_dict == {"u": 0, "v": 0}
assert domain.bound_list[2].condition_dict == {"u": 0, "v": 0}
assert domain.bound_list[3].condition_dict == {"u": 1.0, "v": 0}
assert domain.bound_list[4].condition_dict == {"p": 0}
print("Cavity geometry and boundary conditions OK.")

model0 = df.QCPINN(
    input_vars=["x", "y"],
    output_vars=["u", "v", "p"],
    hidden_layer_pre=QC_PRE,
    hidden_layer_post=QC_POST,
    nqubits=QC_NQUBITS,
    q_layer_iterations=QC_ITERATIONS,
)
n_params = count_params(model0)
assert n_params == 877, f"Expected 877 QCPINN parameters, got {n_params}"
print(f"QCPINN params: {n_params} (expected 877)")

calc_loss = df.calc_loss_simple(domain)
model, model_best = model0.train_adam(
    calc_loss=calc_loss,
    learning_rate=0.004,
    epochs=2,
    threshold_loss=None,
    print_every=1,
)
print(f"Adam done. final loss = {model.loss_history['total_loss'][-1]:.4e}")

area_eval = domain.area_list[0].evaluate(model_best)
area_eval.sampling_area([20, 20])
assert area_eval.data_dict["u"].size == 400

vertical = df.geometry.line_vertical(0.5, [0.0, 1.0]).evaluate(model_best)
vertical.sampling_line(20)
horizontal = df.geometry.line_horizontal(0.5, [0.0, 1.0]).evaluate(model_best)
horizontal.sampling_line(20)
assert vertical.data_dict["u"].size == 20
assert horizontal.data_dict["v"].size == 20
print("Field and centerline evaluation OK.")
print("QCPINN cavity pipeline OK.")
