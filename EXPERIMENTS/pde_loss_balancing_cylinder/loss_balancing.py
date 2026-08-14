"""Standalone per-equation gradient balancing for DeepFlow PDE losses.

This module intentionally lives outside ``src/deepflow``.  It uses the same
batched forward path as ``ProblemDomain._batched_loss`` but keeps PDE equations
separate until their weights have been computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch


def _gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = loss.new_zeros(())
    for grad in grads:
        if grad is not None:
            squared_norm = squared_norm + grad.detach().square().sum()
    return squared_norm.sqrt()


def _last_linear_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    if not linear_layers:
        raise ValueError("The model has no Linear layer for last-layer balancing.")
    return [parameter for parameter in linear_layers[-1].parameters() if parameter.requires_grad]


def _selected_parameters(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    if scope == "full":
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
    if scope == "last_layer":
        return _last_linear_parameters(model)
    raise ValueError("scope must be 'full' or 'last_layer'")


def _loss_components(domain, model) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return BC/IC totals and one scalar MSE per PDE residual equation."""
    losses: dict[str, torch.Tensor | float] = {
        "pde_loss": 0.0,
        "bc_loss": 0.0,
        "ic_loss": 0.0,
    }
    pde_terms: torch.Tensor | None = None

    groups: dict[str, list] = {}
    for geometry in domain:
        groups.setdefault(geometry.physics_type, []).append(geometry)

    for physics_type, geometries in groups.items():
        input_keys = list(geometries[0].inputs_tensor_dict)
        batched_inputs = {
            key: torch.cat([geometry.inputs_tensor_dict[key] for geometry in geometries])
            for key in input_keys
        }
        batched_outputs = model(batched_inputs)

        start = 0
        for geometry in geometries:
            end = start + len(geometry.X_)
            geometry.model_inputs = geometry.inputs_tensor_dict
            geometry.model_outputs = {
                key: values[start:end] for key, values in batched_outputs.items()
            }
            geometry._compute_residual_field()

            term_losses = geometry.residual_field_raw.square().mean(dim=1)
            if physics_type == "PDE":
                if pde_terms is None:
                    pde_terms = term_losses
                elif pde_terms.shape != term_losses.shape:
                    raise ValueError("All PDE geometries must expose the same number of residual equations.")
                else:
                    pde_terms = pde_terms + term_losses
            else:
                losses[f"{physics_type.lower()}_loss"] += term_losses.sum()
            start = end

    if pde_terms is None:
        parameter = next(model.parameters())
        pde_terms = parameter.new_zeros(1)

    losses["pde_loss"] = pde_terms.sum()
    return losses, pde_terms


@dataclass
class GradientBalancedPDELoss:
    """Callable DeepFlow loss with cached, detached gradient-norm weights."""

    domain: object
    scope: str = "full"
    alpha: float = 0.9
    eps: float = 1e-12
    min_weight: float = 0.05
    max_weight: float = 20.0
    step: int = 0
    weights: torch.Tensor | None = None
    diagnostics: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if not 0.0 < self.min_weight <= self.max_weight:
            raise ValueError("weight bounds must satisfy 0 < min_weight <= max_weight")

    def update(self, model: torch.nn.Module, epoch: int) -> None:
        """Update weights once at an explicit optimizer epoch boundary."""
        _, pde_terms = _loss_components(self.domain, model)
        parameters = _selected_parameters(model, self.scope)
        norms = torch.stack([_gradient_norm(term, parameters) for term in pde_terms])

        target = norms.mean()
        instantaneous_weights = target / norms.clamp_min(self.eps)
        instantaneous_weights = instantaneous_weights.clamp(self.min_weight, self.max_weight)
        instantaneous_weights = instantaneous_weights / instantaneous_weights.mean()

        old_weights = (
            torch.ones_like(instantaneous_weights)
            if self.weights is None
            else self.weights.to(instantaneous_weights)
        )
        blended_weights = (
            self.alpha * old_weights
            + (1.0 - self.alpha) * instantaneous_weights
        )
        self.weights = (blended_weights / blended_weights.mean()).detach()

        weighted_norms = self.weights * norms
        self.diagnostics.append(
            {
                "epoch": epoch,
                "term_losses": pde_terms.detach().cpu().tolist(),
                "gradient_norms": norms.detach().cpu().tolist(),
                "instantaneous_weights": instantaneous_weights.detach().cpu().tolist(),
                "weights": self.weights.cpu().tolist(),
                "weighted_gradient_norms": weighted_norms.detach().cpu().tolist(),
            }
        )

    def __call__(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        losses, pde_terms = _loss_components(self.domain, model)
        weights = (
            torch.ones_like(pde_terms)
            if self.weights is None
            else self.weights.to(device=pde_terms.device, dtype=pde_terms.dtype)
        )
        balanced_pde_loss = (weights * pde_terms).sum()
        losses["total_loss"] = losses["bc_loss"] + losses["ic_loss"] + balanced_pde_loss
        self.step += 1
        return losses


def raw_pde_term_losses(domain, model) -> torch.Tensor:
    """Evaluate individual unweighted PDE equation MSEs on current coordinates."""
    _, terms = _loss_components(domain, model)
    return terms.detach()


@dataclass
class FixedWeightedPDELoss:
    """Callable DeepFlow loss with fixed per-equation PDE weights."""

    domain: object
    weights: tuple[float, ...]

    def __call__(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        losses, pde_terms = _loss_components(self.domain, model)
        weights = pde_terms.new_tensor(self.weights)
        if weights.shape != pde_terms.shape:
            raise ValueError(
                f"Expected {pde_terms.numel()} fixed weights, got {weights.numel()}"
            )
        weighted_pde_loss = (weights * pde_terms).sum()
        losses["total_loss"] = losses["bc_loss"] + losses["ic_loss"] + weighted_pde_loss
        return losses
