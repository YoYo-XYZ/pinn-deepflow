# Model Persistence

DeepFlow's restricted PyTorch artifact is the default format for exact `FNN`,
`PINN`, and `RFFPINN` models. Save through the model method and load through
the package function:

```python
import deepflow as df

model.save("trained_model.pt")
restored = df.load_model("trained_model.pt")
```

The version 1 `.pt` artifact is a DeepFlow model artifact, not an exact
training checkpoint or a deployment export. It stores the reconstructible
model configuration, CPU-normalized `state_dict`, loss history, and the
model's `float32` or `float64` dtype. Optimizer state, scheduler state, epoch
state, training samples, PDE definitions, and random state are not stored.

Loading uses the configured DeepFlow device by default. Pass a device to
override it, which is useful when moving an artifact to a CPU-only machine:

```python
restored = df.load_model("trained_model.pt", device="cpu")
```

The loaded model is reconstructed strictly, placed in evaluation mode, and
retains the artifact dtype without changing DeepFlow's global device or dtype.
The `.pt` suffix is added when omitted; another suffix is rejected. Saving
replaces an existing artifact atomically and does not create missing parent
directories.

## Supported models and activations

Restricted persistence supports exact instances of `FNN`, `PINN`, and
`RFFPINN`. The supported activations are `Tanh`, `Sigmoid`, `ReLU`,
`LeakyReLU`, `ELU`, `GELU`, `SiLU`, `Softplus`, and `Identity`. Known
constructor settings such as `inplace`, `negative_slope`, `alpha`,
`approximate`, `beta`, and `threshold` are stored as data.

Custom model subclasses, custom or subclassed activations, callable weight
initializers, active hard constraints, hooks, replaced forward methods, and
other Python behavior are rejected before the destination is opened. Use the
trusted pickle interface for those models instead.

## Compatibility and errors

The loader accepts only the complete version 1 schema. Unknown or unsupported
format versions, missing or unknown fields, invalid configuration values, and
missing, unexpected, or shape-mismatched state entries raise
`ModelPersistenceError`. A format version is a schema gate; package versions
stored in the artifact are diagnostic metadata. When a supported artifact was
written by a different DeepFlow or PyTorch version, loading emits one
`UserWarning` listing each mismatch and then still attempts strict
reconstruction.

Restricted loading always deserializes on CPU with PyTorch's
`weights_only=True` mode before final device placement. It never falls back to
pickle automatically and never treats artifact values as import paths.

## Trusted pickle fallback

Models that require geometry-derived hard constraints or other custom Python
behavior can use the legacy pickle interface explicitly:

```python
model.save_as_pickle("trusted_model.pkl")
restored = df.load_from_pickle("trusted_model.pkl")
```

Pickle is a trusted-file-only fallback: deserialization can execute Python
code. Restricted `.pt` loading reduces executable-deserialization risk through
its data-only schema and restricted loader, but it does not make hostile
artifacts harmless. Do not load either format from an untrusted source.
