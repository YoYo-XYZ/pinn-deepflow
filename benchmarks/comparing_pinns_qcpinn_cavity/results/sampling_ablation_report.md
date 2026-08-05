# PINN Cavity Sampling Ablation

Independent PINN-only tests with the same seed, architecture, CFD reference, and training budget (0 Adam + 50 L-BFGS epochs).

- Architecture: `PINN(width=48, length=4)`
- Runs per variant: 1
- Base seed: 69

| Variant | Final loss | Relative L2 u | Relative L2 v | Relative L2 speed | Relative L2 p | Vertical RMSE | Horizontal RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 4.707765e-02 | 2.393049e-01 | 2.824324e-01 | 1.708541e-01 | 4.609773e-01 | 4.129838e-02 | 1.369620e-02 |
| `corner_excluded` | 3.242075e-02 | 2.836090e-01 | 3.266536e-01 | 1.940943e-01 | 4.885430e-01 | 4.002615e-02 | 1.786198e-02 |
| `boundary_refined` | 3.681243e-02 | 2.040085e-01 | 2.177367e-01 | 1.374256e-01 | 4.064263e-01 | 3.214438e-02 | 9.221125e-03 |

The `corner_excluded` test removes only the endpoint samples from the four rectangle edges. The `boundary_refined` test retains the endpoint behavior but doubles the uniform samples on each physical edge.
