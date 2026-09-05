"""Shared orchestration for steady 2-D flow benchmark variants."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import deepflow as df  # noqa: E402

from .config import BenchmarkConfig  # noqa: E402
from .reporting import (  # noqa: E402
    aggregate_metrics,
    collect_metrics,
    collect_reference_metrics,
    evaluate_area,
    evaluate_line,
    load_model,
    plot_results,
    representative_run_index,
    save_model,
    train_one,
    write_markdown_report,
)


FLOW_VARIANTS = (
    "PINN-UVP",
    "QCPINN-UVP",
    "PINN-PSIP",
    "QCPINN-PSIP",
)
FLOW_FORMULATIONS = {
    "PINN-UVP": "uvp",
    "QCPINN-UVP": "uvp",
    "PINN-PSIP": "psip",
    "QCPINN-PSIP": "psip",
}
PDE_RESIDUAL_COUNTS = {"uvp": 3, "psip": 2}
QC_PRE = [32]
QC_POST = [32]
QC_NQUBITS = 4
QC_ITERATIONS = 10


class FlowBenchmarkHarness:
    """Deep shared interface for flow model, training, and profile runs.

    A suite supplies only its domain/PDE builders and profile geometry. This
    module owns variant validation, model construction, training, evaluation,
    native persistence, plots, aggregation, and report generation.
    """

    def __init__(
        self,
        *,
        problem: str,
        results_dir: Path,
        report_name: str,
        default_config: BenchmarkConfig,
        smoke_config: BenchmarkConfig | None = None,
        domain_builder: Callable[[str, BenchmarkConfig], object],
        pde_builder: Callable[[str], object],
        profile_geometries: Callable[[], Mapping[str, object]],
        profile_fields: Mapping[str, str],
        report_metadata: Callable[[], Mapping[str, Any]] | None = None,
    ):
        self.problem = problem
        self.results_dir = Path(results_dir)
        self.report_name = report_name
        self.default_config = default_config
        self.smoke_config = default_config if smoke_config is None else smoke_config
        self.domain_builder = domain_builder
        self.pde_builder = pde_builder
        self.profile_geometries = profile_geometries
        self.profile_fields = dict(profile_fields)
        self.report_metadata = report_metadata or (lambda: {})
        self.variants = FLOW_VARIANTS
        self.formulations = dict(FLOW_FORMULATIONS)

    def _variant_slug(self, variant: str) -> str:
        return variant.lower().replace("-", "_")

    def _validate_variant(self, variant: str) -> None:
        if variant not in self.variants:
            raise ValueError(
                f"Unknown {self.problem} benchmark variant: {variant!r}"
            )

    def available_variants(self) -> tuple[str, ...]:
        """Return variants whose optional model backends are installed."""
        if hasattr(df, "QCPINN"):
            return self.variants
        return tuple(
            variant
            for variant in self.variants
            if not variant.startswith("QCPINN")
        )

    @staticmethod
    def _output_vars(formulation: str) -> list[str]:
        if formulation == "uvp":
            return ["u", "v", "p"]
        if formulation == "psip":
            return ["psi", "p"]
        raise ValueError(f"Unknown formulation: {formulation!r}")

    def build_pinn_model(
        self,
        formulation: str,
        config: BenchmarkConfig | None = None,
    ):
        """Build the standard PINN for one shared flow formulation."""
        config = self.default_config if config is None else config
        return df.PINN(
            input_vars=["x", "y"],
            output_vars=self._output_vars(formulation),
            width=config.width,
            length=config.depth,
        )

    def build_qcpinn_model(
        self,
        formulation: str,
        config: BenchmarkConfig | None = None,
    ):
        """Build the optional QCPINN for one shared flow formulation."""
        del config
        # FLEX: QCPINN construction requires the optional PennyLane backend.
        try:
            import pennylane  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PennyLane is required for QCPINN variants; install it with "
                "pip install pennylane."
            ) from exc
        try:
            qcpinn = df.QCPINN
        except AttributeError as exc:
            raise RuntimeError("DeepFlow was imported without QCPINN support.") from exc
        return qcpinn(
            input_vars=["x", "y"],
            output_vars=self._output_vars(formulation),
            hidden_layer_pre=QC_PRE,
            hidden_layer_post=QC_POST,
            nqubits=QC_NQUBITS,
            q_layer_iterations=QC_ITERATIONS,
        )

    def build_model(
        self,
        variant: str,
        config: BenchmarkConfig | None = None,
    ):
        """Build a named standard or quantum flow model variant."""
        self._validate_variant(variant)
        formulation = self.formulations[variant]
        if variant.startswith("QCPINN"):
            return self.build_qcpinn_model(formulation, config)
        return self.build_pinn_model(formulation, config)

    def evaluate_profiles(
        self,
        model,
        formulation: str,
        points: int = 50,
    ) -> dict[str, object]:
        """Evaluate suite-specific profiles through the shared line path."""
        return {
            name: evaluate_line(
                geometry,
                model,
                self.pde_builder(formulation),
                points,
            )
            for name, geometry in self.profile_geometries().items()
        }

    def _profile_reference_metrics(self, profiles, reference_solution):
        metrics = {}
        for name, field in self.profile_fields.items():
            values = collect_reference_metrics(
                profiles[name],
                reference_solution,
                field=field,
                final_coordinate=None,
            )
            metrics.update(
                {f"{name}_{key}": value for key, value in values.items()}
            )
        return metrics

    def run_once(
        self,
        variant: str,
        config: BenchmarkConfig,
        reference_solution=None,
        model_factory: Callable[[BenchmarkConfig], object] | None = None,
    ) -> dict:
        """Train, evaluate, and collect metrics for one seeded variant."""
        self._validate_variant(variant)
        formulation = self.formulations[variant]
        domain = self.domain_builder(formulation, config)
        factory = (
            (lambda: model_factory(config))
            if model_factory is not None
            else (lambda: self.build_model(variant, config))
        )
        model, training_info = train_one(domain, factory, config)
        evaluator = evaluate_area(domain, model, list(config.eval_grid))
        profiles = self.evaluate_profiles(model, formulation)
        residual_count = PDE_RESIDUAL_COUNTS[formulation]
        metrics = {
            **training_info,
            **collect_metrics(
                evaluator,
                model,
                reference_solution=reference_solution,
            ),
            "variant": variant,
            "formulation": formulation,
            "model_type": "QCPINN" if variant.startswith("QCPINN") else "PINN",
            "seed": config.seed,
            "pde_residual_count": residual_count,
            "pde_loss_per_equation": (
                training_info["final_pde_loss"] / residual_count
            ),
            # FLEX: parameter counts are benchmark metadata not exposed by the
            # DeepFlow model API.
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
        if reference_solution is not None:
            metrics.update(
                self._profile_reference_metrics(profiles, reference_solution)
            )
        return {
            "variant": variant,
            "formulation": formulation,
            "model": model,
            "domain": domain,
            "evaluator": evaluator,
            "profiles": profiles,
            "metrics": metrics,
        }

    def run_variant(
        self,
        variant: str,
        config: BenchmarkConfig,
        output_dir: Path,
        reference_solution=None,
        model_factory: Callable[[BenchmarkConfig], object] | None = None,
    ) -> dict:
        """Run, persist, plot, and report one model variant."""
        self._validate_variant(variant)
        output_dir = Path(output_dir)
        runs = [
            self.run_once(
                variant,
                replace(config, seed=seed, seeds=[seed]),
                reference_solution=reference_solution,
                model_factory=model_factory,
            )
            for seed in config.seeds
        ]
        representative_index = representative_run_index(
            [run["metrics"] for run in runs]
        )
        representative = runs[representative_index]
        metrics = {
            **representative["metrics"],
            **aggregate_metrics([run["metrics"] for run in runs]),
            "num_runs": len(runs),
            "representative_run_idx": representative_index,
        }
        prefix = f"{self.problem}_{self._variant_slug(variant)}"
        model_path = save_model(representative["model"], output_dir / prefix)
        artifacts = [model_path]
        artifacts.extend(
            plot_results(representative["evaluator"], output_dir, prefix=prefix)
        )
        return {
            **representative,
            "metrics": metrics,
            "artifacts": artifacts,
            "model_path": model_path,
            "runs": runs,
            "representative_run_idx": representative_index,
        }

    def compare_variant(
        self,
        variant: str,
        model_path: Path,
        config: BenchmarkConfig | None = None,
        output_dir: Path | None = None,
        reference_solution=None,
    ) -> dict:
        """Load, evaluate, persist, and report one native model variant."""
        self._validate_variant(variant)
        config = self.default_config if config is None else config
        output_dir = (
            self.results_dir / "comparison"
            if output_dir is None
            else Path(output_dir)
        )
        model_path = Path(model_path)
        if model_path.suffix != ".pkl":
            model_path = Path(f"{model_path}.pkl")
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{self.problem.capitalize()} {variant} model not found: {model_path}"
            )

        formulation = self.formulations[variant]
        model = load_model(model_path)
        domain = self.domain_builder(formulation, config)
        evaluator = evaluate_area(domain, model, list(config.eval_grid))
        profiles = self.evaluate_profiles(model, formulation)
        metrics = {
            **collect_metrics(
                evaluator,
                model,
                reference_solution=reference_solution,
                final_coordinate=None,
            ),
            "variant": variant,
            "formulation": formulation,
            "source_model": str(model_path),
        }
        if reference_solution is not None:
            for field in ("v", "p"):
                if field not in evaluator.data_dict:
                    continue
                values = collect_reference_metrics(
                    evaluator,
                    reference_solution,
                    field=field,
                    final_coordinate=None,
                    center=field == "p",
                )
                metrics.update(
                    {f"{field}_{key}": value for key, value in values.items()}
                )
            metrics.update(
                self._profile_reference_metrics(profiles, reference_solution)
            )

        prefix = f"comparison_{self._variant_slug(variant)}"
        persisted_path = save_model(model, output_dir / model_path.stem)
        artifacts = [persisted_path]
        artifacts.extend(plot_results(evaluator, output_dir, prefix=prefix))
        return {
            "variant": variant,
            "model": model,
            "domain": domain,
            "evaluator": evaluator,
            "profiles": profiles,
            "metrics": metrics,
            "artifacts": artifacts,
            "model_path": persisted_path,
            "source_model_path": model_path,
        }

    def compare_suite(
        self,
        model_paths: Mapping[str, Path],
        config: BenchmarkConfig | None = None,
        output_dir: Path | None = None,
        variants: Sequence[str] | None = None,
        reference_solution=None,
        reference_label: str | None = None,
    ) -> dict:
        """Compare selected native models through the shared flow harness."""
        config = self.default_config if config is None else config
        output_dir = (
            self.results_dir / "comparison"
            if output_dir is None
            else Path(output_dir)
        )
        variants = self.available_variants() if variants is None else tuple(variants)
        if not variants:
            raise ValueError(f"At least one {self.problem} model is required.")
        if len(set(variants)) != len(variants):
            raise ValueError(f"{self.problem.capitalize()} variants must be unique.")
        for variant in variants:
            self._validate_variant(variant)

        results = {
            variant: self.compare_variant(
                variant,
                model_paths[variant],
                config,
                output_dir,
                reference_solution=reference_solution,
            )
            for variant in variants
        }
        report_metrics = {
            "variants": ", ".join(variants),
            "reference": (
                reference_label
                if reference_label is not None
                else "FEM"
                if reference_solution is not None
                else "not requested"
            ),
        }
        artifacts = []
        for variant, result in results.items():
            report_metrics.update(
                {
                    f"{self._variant_slug(variant)}_{key}": value
                    for key, value in result["metrics"].items()
                }
            )
            artifacts.extend(result["artifacts"])
        report = write_markdown_report(
            output_dir / self.report_name,
            f"{self.problem.capitalize()} model comparison",
            config,
            report_metrics,
            artifacts,
        )
        return {"variants": results, "report": report, "artifacts": artifacts}

    def run_suite(
        self,
        config: BenchmarkConfig | None = None,
        output_dir: Path | None = None,
        variants: Sequence[str] | None = None,
        reference_solution=None,
    ) -> dict:
        """Run selected variants through the shared flow harness."""
        config = self.default_config if config is None else config
        output_dir = self.results_dir if output_dir is None else Path(output_dir)
        variants = self.available_variants() if variants is None else tuple(variants)
        if not variants:
            raise ValueError(f"At least one {self.problem} variant is required.")
        if len(set(variants)) != len(variants):
            raise ValueError(f"{self.problem.capitalize()} variants must be unique.")
        for variant in variants:
            self._validate_variant(variant)

        results = {
            variant: self.run_variant(
                variant,
                config,
                output_dir,
                reference_solution=reference_solution,
            )
            for variant in variants
        }
        report_metrics = dict(self.report_metadata())
        report_metrics["variants"] = ", ".join(variants)
        if reference_solution is not None:
            report_metrics["reference_backend"] = getattr(
                reference_solution, "metadata", {}
            ).get("backend", "FEM")
        artifacts = []
        for variant, result in results.items():
            report_metrics.update(
                {
                    f"{self._variant_slug(variant)}_{key}": value
                    for key, value in result["metrics"].items()
                }
            )
            artifacts.extend(result["artifacts"])
        report = write_markdown_report(
            output_dir / self.report_name,
            f"{self.problem.capitalize()} PINN/QCPINN benchmark",
            config,
            report_metrics,
            artifacts,
        )
        return {"variants": results, "report": report, "artifacts": artifacts}

    def run_variant_cli(
        self,
        variant: str,
        model_factory: Callable[[BenchmarkConfig], object] | None = None,
        argv=None,
    ) -> dict:
        """Run one variant with the shared per-cell CLI options."""
        parser = argparse.ArgumentParser(
            description=f"Run the {variant} {self.problem} benchmark."
        )
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--num_runs", type=self._positive_int)
        parser.add_argument("--epochs_adam", type=self._nonnegative_int)
        parser.add_argument("--epochs_lbfgs", type=self._nonnegative_int)
        parser.add_argument("--output-dir", type=Path, default=self.results_dir)
        args = parser.parse_args(argv)
        result = self.run_variant(
            variant,
            self._config_from_args(args),
            args.output_dir,
            model_factory=model_factory,
        )
        metrics = result["metrics"]
        print(
            f"{variant}: final total loss={metrics['final_total_loss']:.6e}, "
            f"PDE residual={metrics.get('mean_abs_continuity_residual', float('nan')):.6e}"
        )
        print(f"Model: {result['model_path']}")
        return result

    def _config_from_args(self, args) -> BenchmarkConfig:
        config = self.default_config
        if args.smoke:
            config = self.smoke_config
        if args.num_runs is not None:
            config = replace(
                config,
                seeds=[config.seed + index for index in range(args.num_runs)],
            )
        if args.epochs_adam is not None:
            config = replace(config, epochs_adam=args.epochs_adam)
        if args.epochs_lbfgs is not None:
            config = replace(config, epochs_lbfgs=args.epochs_lbfgs)
        return config

    @staticmethod
    def _positive_int(value: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError("must be a positive integer") from exc
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return parsed

    @staticmethod
    def _nonnegative_int(value: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "must be a non-negative integer"
            ) from exc
        if parsed < 0:
            raise argparse.ArgumentTypeError("must be a non-negative integer")
        return parsed
