# DeepFlow
[![PyPI version](https://badge.fury.io/py/deepflow.svg)](https://badge.fury.io/py/deepflow)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![alt text](<examples/logo_name_deepflow_d0jn2w1s269cpsiibh89_23 (3) (1).svg>)

DeepFlow is a framework for solving PDEs like Navier-Stokes equations using **Physics-Informed Neural Networks (PINNs)**.

## Features
* **CFD-solver style**: Straightforward workflow similar to commercial CFD software.
* **Physics attached Geometry**: Explicitly attach Physics and PINN models to geometries.
* **Built-in Visualization**: Comes with tools to evaluate and plot results.
* **Flexible Domain Definition**: Easily define complex geometries.
* **GPU accererlation**: Enable GPU for faster training.


![Steady-State Channel Flow Simulation Result](examples/cylinder_flow_steady/cylinder_flow_steady.png)



## Install DeepFlow
```
pip install deepflow
```




## Quick Start

This example shows how to simulate steady channel flow. We recommend using a Python notebook (`.ipynb`) for an interactive run.

### 1. Define the Geometry and Physics

First, define the geometries of your domain. Then, attach the boundary conditions and the governing PDE to the geometric entities.

```python
from deepflow import PINN, Geometry, Physics, NetworkTrainer, Evaluate, ProblemDomain

# Define the area and bound
rectangle = Geometry.rectangle([0,5], [0,1])
bound_list = rectangle.bound_list
area_list = [rectangle]

domain = ProblemDomain(rectangle.bound_list, rectangle, device='cpu')
# Define the physics at the geometry
domain.bound_list[0].define_bc({'u':0,'v':0})
domain.bound_list[1].define_bc({'u':0,'v':0})
domain.bound_list[2].define_bc({'u': 1, 'v': 0})
domain.bound_list[3].define_bc({'p':0})
domain.area_list[0].define_pde(Physics.NVS_nondimensional(U=0.0001, L=1, mu=0.001, rho=1000))

# Sampling initial collocation points
domain.sampling_random_r([100, 100, 200, 100], [5000])
domain.show_coordinates(display_conditions = True) #display collocation points
```

### 2. Define the Model and Loss

Create the PINN model and a function to calculate the loss. This function will handle the random sampling of points for each training step.

```python
# Initialize the PINN model
model0 = PINN(width=40, length=4)

# Design the steps to calculate loss
iterations = 0
def calc_loss(model):
    global iterations
    iterations += 1

    # Add collocation points using based on residual
    if iterations % 500==0:
        domain.sampling_RAR([40, 40, 80, 40], [1000], model)

    # BC Loss
    bc_loss = 0.0
    for bound in domain.bound_list:
        bc_loss += bc.calc_loss(model)

    # PDE Loss
    pde_loss = 0.0
    for area in domain.area_list:
        pde_loss += area.calc_loss(model)

    # Total Loss
    total_loss = bc_loss + pde_loss

    return {"bc_loss": bc_loss, "pde_loss": pde_loss, "total_loss": total_loss} # MUST RETURN IN THIS FORMAT
```

### 3. Train the Model

Train the model using the Adam optimizer.

```python
# Train the model
model1 = NetworkTrainer.train_adam(
    model=model0,
    calc_loss=calc_loss,
    learning_rate=0.001,
    epochs=2000,
    print_every=250,
    thereshold_loss=0.05,
    device='cpu'
)
```
### 4. Visualize the Results
After training, you can easily visualize the flow field and training history.
```python
area_eval = Evaluate(model1, domain.area_list[0])
area_eval.sampling_area(500, 100)
colorplot_area_2d = area_eval.plot_data_on_geometry({'u': 'rainbow'})
loss_history = area_eval.plot_loss_curve(log_scale=True)

```
This will produce a visual representation of the steady-state channel flow and loss curve of trained PINN.


This guide is simplified for demonstration purposes. Explore `examples/` for real use cases including:
- steady cylinder flow
- cavity flow
