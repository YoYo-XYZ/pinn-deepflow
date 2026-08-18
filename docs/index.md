# DeepFlow: Physics-Informed Neural Networks for Fluid Dynamics

[![PyPI version](https://badge.fury.io/py/deepflow.svg)](https://badge.fury.io/py/deepflow)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deepflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![DeepFlow Logo](static/logo_name_deepflow.svg)

DeepFlow is a user-friendly framework for solving PDEs, with a focus on fluid
dynamics including the Navier–Stokes equations, using **Physics-Informed
Neural Networks (PINNs)**. It provides a CFD-solver-style workflow that makes
PINN-based simulations accessible and straightforward.

## Getting started

- [**Quick Start**](quickstart.md) — solve steady channel flow in under 20 lines of code
- [**Installation**](install.md) — pip, the `[cfd]` extra, and GPU setup
- [**Examples**](examples.md) — a gallery of 6 worked cases, from Burgers' equation to transient channel flow
- [**API reference**](reference.md) — full documentation of the `deepflow` package
- [**GitHub repository**](https://github.com/YoYo-XYZ/pinn-deepflow) — source, long-form README, and issue tracker

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
    - transient & steady 2D incompressible Navier-Stokes equations, 2D Fourier Heat equation, Burgers' equation
- **Sampling methods**: Uniform, Random, Latin Hypercube Sampling, RAR-G [[0]](https://arxiv.org/abs/2207.10289), R3 [[1]](https://arxiv.org/abs/2207.02338)
- **2D Geometries**: Custom functions, Rectangle, Circle, Polygon, and combinations & subtractions.
- **Hard Boundary Conditions**: Automatic Hard BC w.r.t. to geometry.
- **Neural Network Architectures**: Fully connected feedforward networks (FNN).
- **Optimizers**: Adam, L-BFGS
- **Backend**: PyTorch, with an optional NGSolve FEM reference backend (`pip install "deepflow[cfd]"`)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/LICENSE) file for details.
