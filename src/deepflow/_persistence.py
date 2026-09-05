"""Restricted, versioned persistence for supported DeepFlow models."""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from contextlib import contextmanager
from importlib import metadata
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import nn

from .nn import FNN, NN, PINN, RFFPINN
from .utility import get_device


class ModelPersistenceError(RuntimeError):
    """Raised when a restricted DeepFlow model artifact is invalid."""


_FORMAT_MARKER = "deepflow.model"
_FORMAT_VERSION = 1
_ARTIFACT_KEYS = frozenset(
    {"format", "version", "versions", "model", "dtype", "state_dict", "loss_history"}
)
_MODEL_KEYS = frozenset({"type", "config"})
_VERSION_KEYS = frozenset({"deepflow", "torch"})
_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_WEIGHT_INIT_NAMES = frozenset({"kaiming", "he", "xavier", "glorot"})


def _raise(message: str, cause: Exception | None = None) -> None:
    error = ModelPersistenceError(message)
    if cause is None:
        raise error
    raise error from cause


def _same_keys(value: Mapping[Any, Any], expected: frozenset[str], label: str) -> bool:
    try:
        keys = set(value.keys())
    except (TypeError, AttributeError) as exc:
        _raise(f"Invalid {label}: keys must be strings", exc)
    if keys != expected:
        _raise(
            f"Invalid {label}: expected fields {sorted(expected)}, "
            f"got {sorted(keys, key=str)}"
        )
    return True


def _require_mapping(value: Any, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        _raise(f"Invalid {label}: expected a mapping")
    return value


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str:
        _raise(f"Invalid {label}: expected a string")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(f"Invalid {label}: expected an integer")
    if minimum is not None and value < minimum:
        _raise(f"Invalid {label}: expected an integer >= {minimum}")
    return value


def _require_finite_real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _raise(f"Invalid {label}: expected a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        _raise(f"Invalid {label}: expected a finite real number", exc)
    if not math.isfinite(result):
        _raise(f"Invalid {label}: expected a finite real number")
    return result


def _require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        _raise(f"Invalid {label}: expected a boolean")
    return value


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(type(item) is str for item in value):
        _raise(f"Invalid {label}: expected a list of strings")
    return list(value)


_ACTIVATION_CLASSES = MappingProxyType(
    {
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "softplus": nn.Softplus,
        "identity": nn.Identity,
    }
)
_ACTIVATION_NAMES = MappingProxyType(
    {activation: name for name, activation in _ACTIVATION_CLASSES.items()}
)
_ACTIVATION_ARGUMENTS = MappingProxyType(
    {
        "tanh": MappingProxyType({}),
        "sigmoid": MappingProxyType({}),
        "relu": MappingProxyType({"inplace": "bool"}),
        "leaky_relu": MappingProxyType(
            {"negative_slope": "real", "inplace": "bool"}
        ),
        "elu": MappingProxyType({"alpha": "real", "inplace": "bool"}),
        "gelu": MappingProxyType({"approximate": "approximate"}),
        "silu": MappingProxyType({"inplace": "bool"}),
        "softplus": MappingProxyType({"beta": "real", "threshold": "real"}),
        "identity": MappingProxyType({}),
    }
)


def _normalize_activation_argument(value: Any, kind: str, label: str) -> Any:
    if kind == "bool":
        return _require_bool(value, label)
    if kind == "real":
        return _require_finite_real(value, label)
    if kind == "approximate":
        value = _require_string(value, label)
        if value not in {"none", "tanh"}:
            _raise(f"Invalid {label}: {value!r}")
        return value
    _raise(f"Unsupported activation argument kind: {kind!r}")


def _activation_kwargs(source: Any, name: str, *, from_module: bool) -> dict[str, Any]:
    argument_specs = _ACTIVATION_ARGUMENTS[name]
    if not from_module:
        source = _require_mapping(source, "activation constructor arguments")
        _same_keys(
            source,
            frozenset(argument_specs),
            "activation constructor arguments",
        )
    return {
        argument: _normalize_activation_argument(
            getattr(source, argument) if from_module else source[argument],
            kind,
            f"activation.{argument}",
        )
        for argument, kind in argument_specs.items()
    }


def _activation_config(activation: nn.Module) -> dict[str, Any]:
    name = _ACTIVATION_NAMES.get(type(activation))
    if name is None:
        _raise(
            "Restricted .pt persistence does not support custom activation "
            f"{type(activation).__name__}; use the trusted pickle interface instead."
        )
    return {"type": name, "kwargs": _activation_kwargs(activation, name, from_module=True)}


def _validate_activation_config(value: Any) -> dict[str, Any]:
    config = _require_mapping(value, "activation configuration")
    _same_keys(config, frozenset({"type", "kwargs"}), "activation configuration")
    name = _require_string(config["type"], "activation type")
    if name not in _ACTIVATION_CLASSES:
        _raise(f"Unsupported activation identifier: {name!r}")

    normalized_kwargs = _activation_kwargs(config["kwargs"], name, from_module=False)
    return {"type": name, "kwargs": normalized_kwargs}


def _activation_from_config(value: Any) -> nn.Module:
    config = _validate_activation_config(value)
    activation_class = _ACTIVATION_CLASSES[config["type"]]
    try:
        return activation_class(**config["kwargs"])
    except Exception as exc:
        _raise("Could not construct the saved activation", exc)


def _weight_init_config(value: Any) -> str | None:
    if value is None:
        return None
    if callable(value):
        _raise(
            "Restricted .pt persistence does not support callable weight "
            "initializers; use the trusted pickle interface instead."
        )
    if type(value) is not str or value.lower() not in _WEIGHT_INIT_NAMES:
        _raise(
            "Restricted .pt persistence supports only the built-in weight "
            "initialization aliases or None."
        )
    return value


def _common_model_config(model: NN) -> dict[str, Any]:
    input_vars = _require_string_list(model.input_keys, "input variables")
    output_vars = _require_string_list(model.output_keys, "output variables")
    if model.input_num != len(input_vars) or model.output_num != len(output_vars):
        _raise("The model's declared input/output dimensions do not match its names")

    return {
        "input_vars": input_vars,
        "output_vars": output_vars,
        "activation": _activation_config(model.activation),
        "weight_init": _weight_init_config(model.weight_init),
    }


def _hidden_layer_config(value: Any) -> list[int]:
    if not isinstance(value, list):
        _raise("The model's hidden_layer must be a list of positive integers")
    return [_require_int(width, "hidden layer width", minimum=1) for width in value]


def _model_spec(model: NN) -> tuple[str, dict[str, Any]]:
    model_type = type(model)
    if model_type not in {FNN, PINN, RFFPINN}:
        _raise(
            "Restricted .pt persistence does not support custom model subclasses "
            "or other model types; expected an exact FNN, PINN, or RFFPINN "
            f"instance, not {model_type.__name__}. Use the trusted pickle "
            "interface instead."
        )

    for module in model.modules():
        for hook_name in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
            "_state_dict_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
        ):
            if getattr(module, hook_name, None):
                _raise(
                    "Restricted .pt persistence does not support Python hooks; "
                    "use the trusted pickle interface instead."
                )
        if "forward" in getattr(module, "__dict__", {}):
            _raise(
                "Restricted .pt persistence does not support replaced Python "
                "forward methods; use the trusted pickle interface instead."
            )

    for constraint_name in ("hard_constraints", "hard_constants"):
        constraints = getattr(model, constraint_name, None)
        if constraints is not None and (
            not isinstance(constraints, Mapping) or bool(constraints)
        ):
            _raise(
                "Restricted .pt persistence does not support active hard "
                "constraints; use the trusted pickle interface instead."
            )

    common = _common_model_config(model)
    if model_type is FNN:
        config = {
            **common,
            "hidden_layer": _hidden_layer_config(model.hidden_layer),
        }
        return "FNN", config

    if model_type is PINN:
        hidden_layer = _hidden_layer_config(model.hidden_layer)
        width = getattr(model, "width", hidden_layer[0] if hidden_layer else None)
        length = getattr(model, "length", len(hidden_layer))
        width = _require_int(width, "width", minimum=1)
        length = _require_int(length, "length", minimum=0)
        if hidden_layer != [width for _ in range(length)]:
            _raise(
                "A PINN's declared width and length do not match its "
                "reconstructible architecture."
            )
        config = {
            **common,
            "width": width,
            "length": length,
        }
        return "PINN", config

    hidden_layer = _hidden_layer_config(model.hidden_layer)
    width = getattr(model, "width", hidden_layer[0] if hidden_layer else None)
    length = getattr(model, "length", len(hidden_layer))
    width = _require_int(width, "width", minimum=1)
    length = _require_int(length, "length", minimum=0)
    if hidden_layer != [width for _ in range(length)]:
        _raise(
            "An RFFPINN's declared width and length do not match its "
            "reconstructible architecture."
        )
    embed_dim = _require_int(model.embed_dim, "embed_dim", minimum=1)
    if embed_dim % 2:
        _raise("Invalid embed_dim: expected a positive even integer")
    alpha = _require_finite_real(model.alpha, "alpha")
    if alpha <= 0:
        _raise("Invalid alpha: expected a positive finite scalar")
    config = {
        **common,
        "width": width,
        "length": length,
        "embed_dim": embed_dim,
        "alpha": alpha,
    }
    return "RFFPINN", config


def _model_config_keys(model_type: str) -> frozenset[str]:
    common = {"input_vars", "output_vars", "activation", "weight_init"}
    if model_type == "FNN":
        return frozenset(common | {"hidden_layer"})
    if model_type == "PINN":
        return frozenset(common | {"width", "length"})
    if model_type == "RFFPINN":
        return frozenset(common | {"width", "length", "embed_dim", "alpha"})
    _raise(f"Unsupported model identifier: {model_type!r}")


def _validate_model_config(model_type: Any, value: Any) -> tuple[str, dict[str, Any]]:
    model_type = _require_string(model_type, "model type")
    config = _require_mapping(value, "model configuration")
    _same_keys(config, _model_config_keys(model_type), "model configuration")

    normalized: dict[str, Any] = {
        "input_vars": _require_string_list(config["input_vars"], "input variables"),
        "output_vars": _require_string_list(config["output_vars"], "output variables"),
        "activation": _validate_activation_config(config["activation"]),
        "weight_init": _weight_init_config(config["weight_init"]),
    }
    if model_type == "FNN":
        normalized["hidden_layer"] = _hidden_layer_config(config["hidden_layer"])
    else:
        normalized["width"] = _require_int(config["width"], "width", minimum=1)
        normalized["length"] = _require_int(config["length"], "length", minimum=0)
        if model_type == "RFFPINN":
            normalized["embed_dim"] = _require_int(
                config["embed_dim"], "embed_dim", minimum=1
            )
            if normalized["embed_dim"] % 2:
                _raise("Invalid embed_dim: expected a positive even integer")
            normalized["alpha"] = _require_finite_real(config["alpha"], "alpha")
            if normalized["alpha"] <= 0:
                _raise("Invalid alpha: expected a positive finite scalar")
    return model_type, normalized


def _history_keys(input_vars: list[str]) -> tuple[str, ...]:
    keys = ["total_loss", "bc_loss", "pde_loss"]
    if "t" in input_vars:
        keys.append("ic_loss")
    return tuple(keys)


def _copy_loss_history(value: Any, expected_keys: tuple[str, ...]) -> dict[str, list[float]]:
    history = _require_mapping(value, "loss history")
    _same_keys(history, frozenset(expected_keys), "loss history")
    copied: dict[str, list[float]] = {}
    for key in expected_keys:
        values = history[key]
        if not isinstance(values, list):
            _raise(f"Invalid loss history for {key!r}: expected a list")
        copied_values: list[float] = []
        for value_item in values:
            if isinstance(value_item, bool) or not isinstance(value_item, Real):
                _raise(
                    f"Invalid loss history for {key!r}: values must be finite numbers"
                )
            try:
                number = float(value_item)
            except (OverflowError, TypeError, ValueError) as exc:
                _raise(
                    f"Invalid loss history for {key!r}: values must be finite numbers",
                    exc,
                )
            if not math.isfinite(number):
                _raise(
                    f"Invalid loss history for {key!r}: values must be finite numbers"
                )
            copied_values.append(number)
        copied[key] = copied_values
    return copied


def _copy_state_to_cpu(model: NN) -> dict[str, torch.Tensor]:
    try:
        state = model.state_dict()
    except Exception as exc:
        _raise("Could not inspect the model state for restricted persistence", exc)
    copied: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or type(value) is not torch.Tensor:
            _raise("The model state must contain only ordinary named tensors")
        try:
            copied[key] = value.detach().clone().cpu()
        except Exception as exc:
            _raise(f"Could not copy model state entry {key!r} to CPU", exc)
    return copied


def _dtype_for_state(state: Mapping[str, torch.Tensor]) -> tuple[torch.dtype, str]:
    floating_dtypes = {value.dtype for value in state.values() if value.is_floating_point()}
    if len(floating_dtypes) != 1:
        _raise("All floating state tensors must use one supported DeepFlow dtype")
    dtype = next(iter(floating_dtypes))
    if dtype not in _DTYPE_NAMES:
        _raise(f"Unsupported model dtype for restricted persistence: {dtype}")
    return dtype, _DTYPE_NAMES[dtype]


def _module_signature(model: nn.Module) -> list[tuple[str, type[nn.Module]]]:
    return [(name, type(module)) for name, module in model.named_modules()]


def _activation_signature(model: nn.Module) -> list[tuple[str, dict[str, Any]]]:
    result = []
    for name, module in model.named_modules():
        if type(module) in _ACTIVATION_NAMES:
            result.append((name, _activation_config(module)))
    return result


def _compare_model_structure(model: NN, expected: NN) -> None:
    if _module_signature(model) != _module_signature(expected):
        _raise(
            "The model network topology or module types changed after construction; "
            "the restricted artifact cannot be reconstructed."
        )
    if _activation_signature(model) != _activation_signature(expected):
        _raise(
            "The model activation configuration does not match its reconstructible "
            "network."
        )
    if getattr(model, "layer_list", None) != getattr(expected, "layer_list", None):
        _raise(
            "The model's declared architecture does not match its network; "
            "the restricted artifact cannot be reconstructed."
        )


def _compare_state_structure(
    state: Mapping[str, torch.Tensor],
    expected_state: Mapping[str, torch.Tensor],
    *,
    require_cpu: bool = False,
) -> None:
    state_keys = set(state.keys())
    expected_keys = set(expected_state.keys())
    missing = sorted(expected_keys - state_keys)
    unexpected = sorted(state_keys - expected_keys)
    if missing or unexpected:
        _raise(
            "Strict state restoration failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in expected_state:
        value = state[key]
        expected_value = expected_state[key]
        if type(value) is not torch.Tensor:
            _raise(f"Invalid state_dict entry {key!r}: expected a tensor")
        if require_cpu and value.device.type != "cpu":
            _raise(f"Invalid state_dict entry {key!r}: expected a CPU tensor")
        if tuple(value.shape) != tuple(expected_value.shape):
            _raise(
                f"Strict state restoration failed for {key!r}: "
                f"expected shape {tuple(expected_value.shape)}, got {tuple(value.shape)}"
            )
        if value.dtype != expected_value.dtype:
            _raise(
                f"Strict state restoration failed for {key!r}: "
                f"expected dtype {expected_value.dtype}, got {value.dtype}"
            )


def _package_versions() -> dict[str, str]:
    try:
        deepflow_version = metadata.version("deepflow")
    except metadata.PackageNotFoundError:
        deepflow_version = "unknown"
    return {"deepflow": str(deepflow_version), "torch": str(torch.__version__)}


@contextmanager
def _preserve_torch_state():
    """Preserve PyTorch RNG streams and default dtype around reconstruction."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    default_dtype = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.set_default_dtype(default_dtype)


def _construct_model(model_type: str, config: Mapping[str, Any]) -> NN:
    activation = _activation_from_config(config["activation"])
    common = {
        "input_vars": list(config["input_vars"]),
        "output_vars": list(config["output_vars"]),
        "activation": activation,
        "weight_init": config["weight_init"],
    }
    try:
        if model_type == "FNN":
            return FNN(hidden_layer=list(config["hidden_layer"]), **common)
        if model_type == "PINN":
            return PINN(
                width=config["width"],
                length=config["length"],
                **common,
            )
        if model_type == "RFFPINN":
            return RFFPINN(
                width=config["width"],
                length=config["length"],
                embed_dim=config["embed_dim"],
                alpha=config["alpha"],
                **common,
            )
    except Exception as exc:
        _raise(f"Could not reconstruct saved {model_type} model", exc)
    _raise(f"Unsupported model identifier: {model_type!r}")


def _artifact_for_model(model: NN) -> dict[str, Any]:
    try:
        model_type, config = _model_spec(model)
    except ModelPersistenceError:
        raise
    except Exception as exc:
        _raise("Could not validate the model for restricted persistence", exc)
    try:
        with _preserve_torch_state():
            expected = _construct_model(model_type, config)
    except ModelPersistenceError:
        raise
    except Exception as exc:
        _raise("Could not validate the model architecture", exc)

    try:
        _compare_model_structure(model, expected)
    except ModelPersistenceError:
        raise
    except Exception as exc:
        _raise("Could not validate the model network structure", exc)
    state = _copy_state_to_cpu(model)
    dtype, _ = _dtype_for_state(state)
    expected.to(dtype=dtype)
    expected_state = expected.state_dict()
    _compare_state_structure(state, expected_state)
    _, dtype_name = _dtype_for_state(state)
    history = _copy_loss_history(
        model.loss_history,
        _history_keys(config["input_vars"]),
    )
    return {
        "format": _FORMAT_MARKER,
        "version": _FORMAT_VERSION,
        "versions": _package_versions(),
        "model": {"type": model_type, "config": config},
        "dtype": dtype_name,
        "state_dict": state,
        "loss_history": history,
    }


def _normalize_path(file_name: Any) -> Path:
    path = Path(file_name)
    if path.suffix == "":
        try:
            return path.with_suffix(".pt")
        except ValueError as exc:
            _raise("A model path must include a file name", exc)
    if path.suffix != ".pt":
        _raise(
            f"Restricted model artifacts must use the .pt suffix, got {path.suffix!r}"
        )
    return path


def save_model(model: NN, file_name: Any) -> None:
    """Save a supported model as a restricted artifact with atomic replacement."""
    path = _normalize_path(file_name)
    artifact = _artifact_for_model(model)

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        temporary_path = Path(temporary_name)
        os.close(descriptor)
        torch.save(artifact, str(temporary_path))
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _validate_artifact(payload: Any) -> tuple[str, dict[str, Any], torch.dtype, dict[str, torch.Tensor], dict[str, list[float]], dict[str, str]]:
    artifact = _require_mapping(payload, "artifact")
    _same_keys(artifact, _ARTIFACT_KEYS, "artifact")
    format_marker = _require_string(artifact["format"], "artifact format marker")
    if format_marker != _FORMAT_MARKER:
        _raise("Invalid DeepFlow model artifact format marker")
    version = artifact["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        _raise("Invalid DeepFlow model artifact format version")
    if version != _FORMAT_VERSION:
        _raise(f"Unsupported DeepFlow model artifact format version: {version}")

    versions = _require_mapping(artifact["versions"], "package versions")
    _same_keys(versions, _VERSION_KEYS, "package versions")
    normalized_versions = {
        key: _require_string(versions[key], f"package version {key}")
        for key in _VERSION_KEYS
    }

    model = _require_mapping(artifact["model"], "model descriptor")
    _same_keys(model, _MODEL_KEYS, "model descriptor")
    model_type, config = _validate_model_config(model["type"], model["config"])

    dtype_name = _require_string(artifact["dtype"], "model dtype")
    if dtype_name not in _NAME_TO_DTYPE:
        _raise(f"Unsupported model dtype in artifact: {dtype_name!r}")
    dtype = _NAME_TO_DTYPE[dtype_name]

    state_value = _require_mapping(artifact["state_dict"], "state_dict")
    state: dict[str, torch.Tensor] = {}
    for key, value in state_value.items():
        if not isinstance(key, str) or type(value) is not torch.Tensor:
            _raise("Invalid state_dict: expected string keys and ordinary tensors")
        if value.device.type != "cpu":
            _raise("Invalid state_dict: all artifact tensors must be on CPU")
        state[key] = value
    _, state_dtype_name = _dtype_for_state(state)
    if state_dtype_name != dtype_name:
        _raise(
            "Invalid state_dict: floating tensor dtype does not match the recorded model dtype"
        )

    history = _copy_loss_history(
        artifact["loss_history"],
        _history_keys(config["input_vars"]),
    )
    return model_type, config, dtype, state, history, normalized_versions


def _warn_on_version_mismatch(saved: Mapping[str, str]) -> None:
    current = _package_versions()
    mismatches = [
        f"{name}: artifact={saved[name]!r}, running={current[name]!r}"
        for name in ("deepflow", "torch")
        if saved[name] != current[name]
    ]
    if mismatches:
        warnings.warn(
            "DeepFlow model artifact package version mismatch (" + "; ".join(mismatches) + ")",
            UserWarning,
            stacklevel=2,
        )


def load_model(file_name: Any, *, device: Any = None) -> NN:
    """Load a restricted DeepFlow model from a CPU-normalized `.pt` artifact.

    Args:
        file_name: Artifact path. The `.pt` suffix is added when omitted.
        device: Optional target device. If omitted, use DeepFlow's configured
            device.

    Returns:
        A reconstructed model in evaluation mode with restored weights,
        configuration, loss history, and dtype.

    Raises:
        ModelPersistenceError: If the artifact cannot be safely validated or
            reconstructed.
    """
    path = _normalize_path(file_name)
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except OSError:
        raise
    except Exception as exc:
        _raise("Could not deserialize the restricted DeepFlow model artifact", exc)

    model_type, config, dtype, state, history, versions = _validate_artifact(payload)
    _warn_on_version_mismatch(versions)

    try:
        with _preserve_torch_state():
            model = _construct_model(model_type, config)
            model.to(dtype=dtype)
            _compare_state_structure(state, model.state_dict(), require_cpu=True)
            model.load_state_dict(state, strict=True)
    except ModelPersistenceError:
        raise
    except Exception as exc:
        _raise("Could not strictly reconstruct the restricted DeepFlow model", exc)

    model.loss_history = {key: list(values) for key, values in history.items()}
    model.eval()
    try:
        target_device = torch.device(get_device() if device is None else device)
        model.to(target_device)
    except Exception as exc:
        target = get_device() if device is None else device
        _raise(f"Could not place the loaded model on device {target!r}", exc)
    return model
