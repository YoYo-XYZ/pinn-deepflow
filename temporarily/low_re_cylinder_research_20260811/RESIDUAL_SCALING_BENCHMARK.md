# Continuity x30 versus momentum /30

## Result

Multiplying continuity by 30 is **not** a drop-in replacement for dividing both momentum residuals by 30 when DeepFlow's total loss is left unchanged. Across three matched seeds at `mu=0.2`, the raw substitution collapsed the flow.

| Residual scaling | Relative L2 speed | Relative L2 pressure | Final BC loss |
|---|---:|---:|---:|
| Momentum /30 | 6.06% +/- 0.50% | 6.97% +/- 0.10% | 0.00155 +/- 0.00045 |
| Continuity x30 | 91.00% +/- 0.71% | 100.19% +/- 0.18% | 0.25995 +/- 0.00427 |

Values are mean +/- sample standard deviation for seeds 69, 70, and 71. Each run used the same 5x32 tanh network, 300 LHS samples per boundary, 3,000 interior samples, pressure scale 30 with exact outlet pressure, normalized coordinates, outlet velocity-gradient conditions, Adam(600) with BC weight 10, and L-BFGS(50). The same FEM reference was used for validation.

## Why

Let the raw residuals be `c`, `mx`, and `my`. The first formulation uses

`A = (c, mx/30, my/30)`

and the proposed formulation uses

`B = (30c, mx, my) = 30 A`.

Consequently,

`PDE_loss_B = 900 PDE_loss_A`.

The two formulations have the same relative weighting **inside the PDE system**, but not relative to the boundary loss. With DeepFlow's unchanged total loss, continuity x30 makes the PDE term 900 times stronger, allowing the optimizer to preserve a very small PDE residual while accepting a large inlet/wall BC error and a collapsed solution.

## Equivalent alternative

Continuity x30 becomes equivalent if the aggregate PDE loss is also weighted by `1/900` in both Adam and L-BFGS:

`BC_loss + PDE_loss_B/900 = BC_loss + PDE_loss_A`.

A seed-69 control using continuity x30 and PDE weight 1/900 recovered 6.12% speed error and 7.35% pressure error, close to the momentum-/30 benchmark. Small differences arise from finite-precision optimization and stopping behavior.

Therefore:

- `continuity x30` alone: **no**;
- `continuity x30` plus `pde_weight=1/900`: **yes, mathematically equivalent**;
- `momentum /30`: clearer and less likely to create an accidental global loss-scale change.

Complete numeric results are in `residual_scaling_benchmark.json`; the comparison plot is `residual_scaling_benchmark.png`.
