# DeepFlow: Physics-Informed Neural Networks for Fluid Dynamics

[![PyPI version](https://badge.fury.io/py/deepflow.svg)](https://badge.fury.io/py/deepflow)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deepflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-DeepFlow-blue.svg)](https://yoyo-xyz.github.io/pinn-deepflow/)
![DeepFlow Logo](static/logo_name_deepflow.svg)

DeepFlow is a user-friendly framework for solving PDEs, with a focus on fluid dynamics including the Navier–Stokes equations, using **Physics-Informed Neural Networks (PINNs)**. It provides a CFD-solver-style workflow that makes PINN-based simulations accessible and straightforward.

## Table of Contents

- [**Key Features**](#features)
- [Current Implementations](#current-implementations)
- [**Installation**](#installation)
- [Requirements](#requirements)
- [**Quick Start**](#quick-start)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [**DeepFlow Milestones**](#future-milestones)

## Key Features

![promo](static/promo.png)

- ⟁ **Physics-Attached Geometry**: **AUTO GENERATE TRAINING DATA** by explicitly attach physics and neural network to geometries.
- 🔧 **CFD-Solver Style**: Straightforward workflow similar to CFD software.
- 📊 **Built-in Visualization**: Tools to evaluate and plot results.
- 🚀 **GPU Acceleration**: Enable GPU for faster training.
- 🔢 **FP64 Precision**: Switch to double precision for improved PINN accuracy/stability.
- **Flexible Domain Definition**: Easily define complex 2D geometries.

## Current Implementations

- **Supported problems**: solving **forward** partial differential equations (PDEs)
  - transient & steady 2D imcompressible Navier-Stokes equations, 2D Fourier Heat equation, Burgers' equation
- **Sampling methods**: Uniform, Random, Latin Hypercube Sampling, RAR-G [[0]](https://arxiv.org/abs/2207.10289), R3 [[1]](https://arxiv.org/abs/2207.02338)
- **2D Geometries**: Custom functions, Rectangle, Circle, Polygon, and combinations & subtractions.
- **Hard Constraints**: Explicit constant hard constraints are applied
  automatically during domain loss evaluation for supported geometries.
- **Neural Network Architectures**: Fully connected feedforward networks (FNN).
- **Optimizers**: Adam, L-BFGS
- **Backend**: PyTorch
- **Optional FEM reference backend**: NGSolve/Netgen via `pip install "deepflow[cfd]"`
- **Experimental quantum models**: QPINN/QCPINN via a separate PennyLane install

## Installation

You can install DeepFlow via pip:

```bash
pip install deepflow
```

NGSolve reference solutions are optional and are loaded only when used.  In a
supported Python 3.10+ environment, install them with:

```bash
pip install "deepflow[cfd]"
```

For development or to build from source:

```bash
git clone https://github.com/YoYo-XYZ/pinn-deepflow.git
cd pinn-deepflow
pip install -e .
```

## Requirements

- Python >= 3.10
- PyTorch >= 1.7.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- SymPy >= 1.5.0
- SciPy >= 1.5.0
- ultraplot >= 1.0.0

## Quick Start

![](static/deepflow_workflow.svg)
This example demonstrates how to simulate Steady channel flow **under 20 lines of code!** We recommend using a Python notebook (`.ipynb`) for interactive experience.

### 1. Define the Geometry and Physics

```python
import deepflow as df

# Define the area and bounds
rectangle = df.geometry.rectangle([0, 5], [0, 1])
domain = df.domain(rectangle)

domain.show_setup() # Display the domain setup
```

![alt text](static/quickstart/setup_show.png)

```python
# Define Boundary Conditions
domain.bound_list[0].define_bc({'u': 1, 'v': 0})  # Inflow: u=1
domain.bound_list[1].define_bc({'u': 0, 'v': 0})  # Wall: No slip
domain.bound_list[2].define_bc({'p': 0})          # Outflow: p=0
domain.bound_list[3].define_bc({'u': 0, 'v': 0})  # Wall: No slip

# Define PDE (Navier-Stokes)
domain.area_list[0].define_pde(df.pde.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000))

domain.show_setup() # Display the domain setup
```

![alt text](static/quickstart/cond_show.png)

```python
# Sample points: [Left, Bottom, Right, Top], [Interior]
domain.sampling_random([200, 400, 200, 400], [5000])
domain.show_coordinates(display_physics=True)
```

![alt text](static/quickstart/coord_show.png)

### 2. Create and Train the model

```python
# Initialize the PINN model
model0 = df.PINN(width=40, length=4)
```

```python
# Train the model using Adam Optimizer
model1, model1_best = model0.train_adam(
    calc_loss=df.calc_loss_simple(domain),
    learning_rate=0.001,
    epochs=2000,
)
```

### 3. Visualize Results

```python
# Evaluate the best model
prediction = domain.area_list[0].evaluate(model1_best)
prediction.sampling_area([500, 100])

# Plot Velocity Field
_ = prediction.plot_color('u', cmap='jet')

# Plot Training Loss
_ =prediction.plot_loss_curve()
```

![alt text](static/quickstart/flow_field.png)
![alt text](static/quickstart/loss_curve.png)

### Using FP64 (Double) Precision

Recent PINN research shows that training in double precision (FP64) can
significantly improve convergence and accuracy. DeepFlow lets you switch the
entire pipeline—model weights, sampled coordinates, and PDE residuals—to FP64
with one line:

```python
import torch
import deepflow as df

# Enable FP64 globally before defining geometry / model
df.dtype = torch.float64

# ... define geometry, PDE, sample, build model, and train as usual
```

To switch back to the default FP32 precision:

```python
df.dtype = torch.float32
```

Only `torch.float32` and `torch.float64` are supported.

## Examples

Explore the [examples](examples)
 directory for real use cases, including

Steady-state:

- [Steady flow around a cylinder](examples/cylinder_flow_steady)
- [Lid-driven cavity flow](examples/cavity_flow_steady)
- [Backward-facing step flow](examples/BFS_flow_steady)
- [Burgers&#39; Equation](examples/burgers_eq)

Time-dependent:

- [Transient channel flow](examples/channel_flow_transient)
- [Fourier Heat Equation](examples/heat_eq)

## Contributing

Feel free to submit a Pull Request. For major changes, open an issue first to discuss the proposed changes.

## DeepFlow Milestones

1. Define custom PDE
2. Inverse problems PDE
3. 3D Geometries
4. More sampling methods
5. More neural network architectures (e.g., CNN, RNN)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
