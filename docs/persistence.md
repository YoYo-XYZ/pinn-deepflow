# Model Persistence

DeepFlow's restricted PyTorch artifact is the default format for supported
`FNN`, `PINN`, and `RFFPINN` models. It stores the model configuration, weights,
loss history, and floating-point dtype in a versioned `.pt` file.

```python
import deepflow as df

model.save("trained_model.pt")
restored = df.load_model("trained_model.pt")
```

The loader uses the configured DeepFlow device by default. Pass a device to
override it, which is useful when moving an artifact to a CPU-only machine:

```python
restored = df.load_model("trained_model.pt", device="cpu")
```

Restricted persistence supports the built-in activations and initialization
schemes used by the three built-in model classes. It rejects custom model
subclasses, custom activations, callable initializers, active hard constraints,
and other Python behavior before writing a file. It does not create missing
parent directories.

For models that require geometry-derived hard constraints or other custom
Python behavior, use the existing pickle interface explicitly:

```python
model.save_as_pickle("trusted_model.pkl")
restored = df.load_from_pickle("trusted_model.pkl")
```

Only load pickle files from trusted sources: pickle deserialization can execute
Python code. Restricted `.pt` loading reduces executable-deserialization risk
through its data-only schema and restricted loader, but it does not make
hostile artifacts harmless.
