# API Reference

This reference is generated from the public Google-style docstrings in the
DeepFlow source code. Private implementation helpers are intentionally omitted.

## Geometry

::: deepflow.geometry
    options:
      members:
        - custom_data
        - CustomData
        - Bound
        - Area
        - circle
        - rectangle
        - line_horizontal
        - line_vertical
        - line
        - polygon
        - curve
        - point

## Domains and Losses

::: deepflow.domain
    options:
      members:
        - domain
        - ProblemDomain
        - calc_loss_simple
        - calc_loss_weighted

## Physics Attachments

::: deepflow.physicsinformed
    options:
      members:
        - PhysicsAttach
        - function
        - func
        - parabolic_func
        - parabolic

## PDEs

::: deepflow.pde
    options:
      members:
        - PDE
        - CustomPDE
        - NavierStokes
        - StreamFunctionNavierStokes
        - HeatEquation
        - WaveEquation
        - BurgersEquation1D

## Neural Networks

::: deepflow.nn
    options:
      members:
        - HardConstraint
        - hard_constraint
        - NN
        - FNN
        - PINN
        - RFFPINN
        - load_from_pickle

## Model Persistence

::: deepflow._persistence
    options:
      members:
        - ModelPersistenceError
        - load_model

## Evaluation

::: deepflow.evaluation
    options:
      members:
        - Evaluator
        - ReferenceEvaluator
        - GroupEvaluator
        - ReferenceGroupEvaluator

## Visualization

::: deepflow.visualization
    options:
      members:
        - Visualizer

## Utilities

::: deepflow.utility
    options:
      members:
        - latin_hypercube_sampling
        - get_device
        - get_dtype
        - set_dtype
        - manual_seed
        - calc_grad
        - calc_grads
        - to_require_grad
        - torch_to_numpy

## FEM Backend

::: deepflow.fem
    options:
      members:
        - solve_fem

## Reference Solutions

::: deepflow.reference
    options:
      members:
        - ReferenceConfigurationError
        - ReferenceGeometryError
        - ReferenceSolution
        - ReferenceSolver
        - UnsupportedReferencePDE

## Quantum Models (Experimental)

QPINN and QCPINN remain experimental in DeepFlow 0.1.3 and are not part of
the stable API support commitment. They require PennyLane separately and are
intentionally omitted from this stable API reference.
