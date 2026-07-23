"""Shared utilities for initialization-comparison benchmarks."""

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# DeepFlow's evaluator uses the Matplotlib-compatible plotting API. Some
# environments expose an incompatible ``ultraplot`` stub, so use Matplotlib
# for these standalone benchmarks when the required API is absent.
try:
    import ultraplot
except ImportError:
    pass
else:
    if not hasattr(ultraplot, "Figure"):
        sys.modules["ultraplot"] = plt

INITIALIZATIONS = (("Kaiming-uniform", "kaiming"), ("Glorot-normal", "glorot"))
INITIALIZATION_ROLES = {
    "Kaiming-uniform": "DeepFlow current default",
    "Glorot-normal": "alternative",
}


def add_project_src(script_file):
    """Make the script directory and local DeepFlow source importable."""
    script_path = Path(script_file).resolve()
    paths = (script_path.parent, script_path.parents[3] / "src")
    for path in paths:
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


def train_one(
    df,
    build_domain,
    init_name,
    weight_init,
    seed,
    input_vars,
    output_vars,
    width,
    depth,
    learning_rate,
    epochs,
    eval_grid,
    residual_keys,
    field_names,
):
    """Train one initialization and return the benchmark result dictionary."""
    print(f"\n--- Training with {init_name} initialization ---")
    # Reset before each run so both initializers receive the same sampled
    # training domain and the same random starting point.
    df.manual_seed(seed)

    domain = build_domain()
    model = df.PINN(
        width=width,
        length=depth,
        input_vars=input_vars,
        output_vars=output_vars,
        weight_init=weight_init,
    )
    start = time.perf_counter()
    _, best_model = model.train_adam(
        calc_loss=df.calc_loss_simple(domain),
        learning_rate=learning_rate,
        epochs=epochs,
    )
    train_time = time.perf_counter() - start

    # Recompute losses for the returned best model on the fixed training
    # domain. This must happen before sampling the separate evaluation grid,
    # which replaces the area's current coordinates.
    best_model.eval()
    best_loss = df.calc_loss_simple(domain)(best_model)

    prediction = domain.area_list[0].evaluate(best_model)
    prediction.sampling_area(eval_grid)
    data = prediction.data_dict

    result = {
        "init": init_name,
        "time": train_time,
        "seed": seed,
        "final_total": float(best_loss["total_loss"].detach().cpu().item()),
        "final_bc": float(best_loss["bc_loss"].detach().cpu().item()),
        "final_pde": float(best_loss["pde_loss"].detach().cpu().item()),
    }
    result.update(
        {
            result_key: float(np.max(np.abs(data[data_key])))
            for result_key, data_key in residual_keys.items()
        }
    )
    result.update({name: np.asarray(data[name]) for name in ("x", "y", *field_names)})
    return result


def run_comparison(
    df,
    build_domain,
    *,
    seed,
    input_vars,
    output_vars,
    width,
    depth,
    learning_rate,
    epochs,
    eval_grid,
    residual_keys,
    field_names,
):
    """Run the standard Kaiming-default-versus-Glorot comparison."""
    return [
        train_one(
            df,
            build_domain,
            init_name,
            weight_init,
            seed,
            input_vars,
            output_vars,
            width,
            depth,
            learning_rate,
            epochs,
            eval_grid,
            residual_keys,
            field_names,
        )
        for init_name, weight_init in INITIALIZATIONS
    ]


def print_summary(results, metrics, metadata=None):
    """Print the comparison table and the protocol limitations."""
    first, second = results
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(
        "Initializer roles : "
        f"{first['init']} = {INITIALIZATION_ROLES.get(first['init'], 'comparison arm')}; "
        f"{second['init']} = {INITIALIZATION_ROLES.get(second['init'], 'comparison arm')}"
    )
    if metadata:
        for label, value in metadata.items():
            print(f"{label:<19}: {value}")
    print(
        "Statistical limit : one paired seed only; deltas are descriptive and "
        "do not estimate across-seed variation."
    )
    print(f"{'Metric':<20} {first['init']:>18} {second['init']:>18} {'Delta':>18}")
    print("-" * 80)
    for key, label in metrics:
        first_value = first[key]
        second_value = second[key]
        delta = (first_value - second_value) / second_value * 100 if second_value else 0
        print(f"{label:<20} {first_value:>18.6e} {second_value:>18.6e} {delta:>17.1f}%")
    print("=" * 80)


def save_field_plot(results, field_name, output_path, message):
    """Save a side-by-side scatter plot using one color scale."""
    fig, axes = plt.subplots(1, len(results), figsize=(14, 4))
    axes = np.atleast_1d(axes)
    value_range = shared_range(*(result[field_name] for result in results))
    for axis, result in zip(axes, results):
        scatter = axis.scatter(
            result["x"], result["y"], c=result[field_name], s=1, cmap="jet", marker="s"
        )
        if value_range is not None:
            scatter.set_clim(value_range)
        axis.set_title(f"{field_name} – {result['init']}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(scatter, ax=axis, shrink=0.8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"{message} saved to: {output_path}")


def shared_range(*arrays):
    """Return a common finite range for the supplied arrays."""
    values = [np.asarray(array).ravel() for array in arrays if array is not None]
    values = [value for value in values if value.size]
    if not values:
        return None
    values = np.concatenate(values)
    if not np.isfinite(values).any():
        return None
    return float(np.nanmin(values)), float(np.nanmax(values))
