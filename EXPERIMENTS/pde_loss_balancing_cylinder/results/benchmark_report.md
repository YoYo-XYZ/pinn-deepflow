# Cylinder-flow PDE loss-balancing benchmark

Re = 5 (`U=1`, `mu=0.2`, `rho=1`, `L=1`), Adam epochs = 10000, points = 6x200 boundary + 1000 interior.
Balancer: scope = `full`, update after every 500 optimizer epochs, weight smoothing alpha = 0.9.
FEM: NGSolve, mesh size 0.05, 2564 elements, final nonlinear residual 1.630e-10, pressure gauge `configured Dirichlet`.

All PDE losses below are raw, unweighted losses evaluated on a fresh paired collocation set.

| Method | Eval PDE loss | Continuity MSE | X-momentum MSE | Y-momentum MSE | Time (s) |
|---|---:|---:|---:|---:|---:|
| default | 0.0460991 | 0.0143383 | 0.0203603 | 0.0114005 | 271.57 |
| fixed | 1250 | 4.88203 | 357.546 | 887.57 | 247.92 |
| balanced | 0.0421939 | 0.0284624 | 0.00470044 | 0.00903111 | 281.60 |

Fixed/default evaluation PDE-loss ratio: **27115.470**.
Balanced/default evaluation PDE-loss ratio: **0.915**.
Balanced/default training-time ratio: **1.04x**.

## FEM solution errors

| Method | u rel-L2 | v rel-L2 | p rel-L2 | |V| rel-L2 | u MAE | v MAE | p MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 0.962954 | 1.08593 | 1.04014 | 0.912346 | 0.631572 | 0.0659624 | 8.01762 |
| fixed | 0.95071 | 1.1978 | 1.05329 | 0.917714 | 0.620547 | 0.0676654 | 8.03037 |
| balanced | 0.970603 | 1.07046 | 1.0426 | 0.905768 | 0.636688 | 0.0659146 | 8.05092 |

At each update, applied weights satisfy `lambda_new = 0.9 * lambda_old + 0.1 * lambda_hat_new`, followed by mean-one normalization. Weights are detached and frozen between the stated optimizer-epoch boundaries.
