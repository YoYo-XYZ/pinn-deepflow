# DeepFlow: Physics-Informed Neural Networks for Fluid Dynamics

[![PyPI version](https://badge.fury.io/py/deepflow.svg)](https://badge.fury.io/py/deepflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

![DeepFlow Logo](static/logo_name_deepflow.svg)

DeepFlow is a user-friendly framework for solving partial differential equations (PDEs), such as the Navier-Stokes equations, using **Physics-Informed Neural Networks (PINNs)**. It provides a CFD-solver-style workflow to make PINN-based simulations accessible and straightforward.

## Table of Contents

- [**Features**](#features)
- [**Installation**](#installation)
- [Requirements](#requirements)
- [**Quick Start**](#quick-start)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [**DeepFlow Milestones**](#future-milestones)

## Features

- 🔧 **CFD-Solver Style**: Straightforward workflow similar to commercial CFD software.

- 📊 **Built-in Visualization**: Tools to evaluate and plot results.
- 🚀 **GPU Acceleration**: Enable GPU for faster training.
- **Flexible Domain Definition**: Easily define complex geometries.
- **Physics-Attached Geometry**: Explicitly attach physics and PINN models to geometries.

## Installation

You can install DeepFlow via pip:

```bash
pip install deepflow
```

For development or to build from source:

```bash
git clone https://github.com/YoYo-XYZ/deepflow.git
cd deepflow
pip install -e .
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.7.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- SymPy >= 1.5.0

## Quick Start

This example demonstrates how to simulate steady channel flow under 20 lines of code! We recommend using a Python notebook (`.ipynb`) for an interactive experience.

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
domain.show_coordinates(display_conditions=True)
```
![alt text](static/quickstart/coord_show.png)
### 2. Create and Train the model


```python
# Initialize the PINN model
model0 = df.PINN(width=40, length=4)
```
```python
# Train the model using Adam Optimizer
model1 = NetworkTrainer.train_adam(
    model=model0,
    calc_loss=df.calc_loss_simple(domain),
    learning_rate=0.001,
    epochs=2000,
    print_every=250)
```

### 3. Visualize Results
```python
# Evaluate the best model
prediction = domain.area_list[0].evaluate(model1_best)
prediction.sampling_area([500, 100])

# Plot Velocity Field
_ = prediction.plot_color({'u': 'rainbow'})

# Plot Training Loss
_ =prediction.plot_loss_curve(log_scale=True)
```
![alt text](static/quickstart/flow_field.png)
![alt text](static/quickstart/loss_curve.png)

## Examples

Explore the [examples](examples)
 directory for real-world use cases, including:

- [Steady cylinder flow](examples/steady_cylinder_flow/steady_cylinder_flow.ipynb)

Each example includes Jupyter notebooks and data files.

## Contributing

Feel free to submit a Pull Request. For major changes, open an issue first to discuss the proposed changes.

## DeepFlow Milestones

1. Complete support for **time-dependent solutions**
2. Ability to define **custom PDEs**
3. Enhanced customization options in **visualization tools**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
