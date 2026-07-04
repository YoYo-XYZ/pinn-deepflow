# Why DeepFlow Has Lower Total Loss but Localized High Loss at the Inlet Corner

## TL;DR

**Previous claim was WRONG.** DeepFlow does NOT pool all boundary conditions into one MSE. It computes a **sum of per-geometry MSEs** — each boundary gets its own denominator. The inlet per-point weight is `1/100`, identical to DeepXDE.

The actual loss-structure difference is smaller than claimed: DeepFlow splits the two walls into separate MSE terms (500 points each), while DeepXDE combines them into one (~1014 points). This gives DeepFlow **2× more wall weight per point** and an inlet-to-wall ratio of **5× vs DeepXDE's 10×**. Combined with DeepXDE's Hammersley (quasi-random) interior sampling and Glorot-normal initialization, these factors explain the smoother corner behavior in DeepXDE.

---

## 1. The symptom

From the channel-flow benchmark (`benchmarks/channel_flow_2d/results/`):

| Metric | DeepFlow | DeepXDE |
|---|---|---|
| Final total loss | 8.40e-03 | 1.80e-02 |
| Max `|continuity|` | 4.42 | 0.447 |
| Max `|x-momentum|` | 4.97 | 0.339 |

The DeepFlow maxima are localized at the inlet-wall corner, e.g. `(0.05, 0.0)`.

- Top 1% of points contribute **95.7%** of the x-momentum MSE.
- Top 1% of points contribute **97.4%** of the continuity MSE.

DeepFlow's total loss is lower because the bulk of the domain is accurate, but the corner is not.

---

## 2. Corrected loss-formula analysis

### What the code actually does

`_batched_loss` in `domain.py` iterates over each geometry and accumulates per-geometry MSE:

```python
for g in geometries:
    ...
    loss_dict[f'{physics_type.lower()}_loss'] += torch.mean(
        g.residual_field_raw.square().sum(dim=0)
    )
```

So the BC loss is:

```
bc_loss = MSE(inlet,  100 pts) + MSE(bottom, 500 pts)
        + MSE(right,  100 pts) + MSE(top,    500 pts)
```

**Each boundary gets its own denominator** (its own point count), not a shared pool of 1200.

### Empirical verification

`verify_loss_formula.py` confirms this with a random model:

| Interpretation | Value |
|---|---|
| Sum of per-geometry MSEs | **1.649162** |
| Pooled MSE over 1200 points (wrong claim) | 0.162415 |
| `calc_loss_simple` actual output | **1.649162** ✓ |

The actual `bc_loss` matches the per-geometry sum, NOT the pooled MSE.

### Correct per-point weights

| Point type | DeepFlow weight | DeepXDE weight |
|---|---|---|
| Inlet point | 1/100 = **0.010** | 1/109 ≈ **0.009** |
| Wall point (per side) | 1/500 = **0.002** | — |
| Wall point (combined) | — | 1/1014 ≈ **0.001** |

- **Inlet weight: SAME** in both frameworks (~0.01).
- **Wall weight: DeepFlow is 2× higher** (0.002 vs 0.001) because DeepFlow splits top/bottom into separate MSE terms while DeepXDE combines them.
- **Inlet-to-wall ratio**: DeepFlow = 5×, DeepXDE = 10×.

### What was wrong in the previous diagnosis

| Previous claim | Status |
|---|---|
| "DeepFlow pools all BCs into one MSE (1/1200)" | **FALSE** — per-geometry MSE summed |
| "Inlet per-point weight = 1/1200 = 0.00083" | **FALSE** — actual weight = 1/100 = 0.01 |
| "Inlet is 12× under-weighted vs DeepXDE" | **FALSE** — inlet weight is the same |
| "Per-boundary loss terms (highest impact fix)" | **Already done** — DeepFlow already uses per-geometry MSE |

---

## 3. Actual differences between DeepFlow and DeepXDE

### 3a. Wall loss weighting (2× difference)

DeepFlow's `bound_list` treats top and bottom walls as **separate geometries**, each with 500 points and its own MSE term. DeepXDE's `boundary_wall` function selects **both** walls, combining ~1014 points into one MSE.

- DeepFlow wall loss = `S_bottom/500 + S_top/500 = S_walls/500`
- DeepXDE wall loss = `S_walls/1014`

DeepFlow weights each wall point ~2× more. This pushes the network harder toward `u=0, v=0` on walls, potentially sharpening the corner transition and increasing PDE residuals there.

### 3b. Interior point distribution (Hammersley vs random)

DeepXDE defaults to `train_distribution='Hammersley'` — a quasi-random low-discrepancy sequence with better space-filling coverage. DeepFlow uses pure random sampling.

Hammersley points provide more uniform coverage near corners and boundaries, which helps the network learn smoother transitions in those regions.

### 3c. Network initialization

- DeepFlow: PyTorch default `nn.Linear` initialization (Kaiming uniform)
- DeepXDE: `"Glorot normal"` (Xavier normal)

Different initializations can lead to different convergence basins, especially for problems with corner singularities.

### 3d. Boundary sampling geometry

- DeepFlow: each side sampled independently (corners approached from both sides)
- DeepXDE: perimeter sampled as a continuous curve (corners are measure-zero)

With random sampling, exact corner coincidences are near-zero in both, but DeepFlow's independent per-side sampling places conflicting-label points closer together near corners.

---

## 4. Physical interpretation

The benchmark uses a discontinuous inlet profile: `u=1` at the inlet and `u=0` at the walls immediately above and below. This is a corner singularity. The network cannot represent an exact discontinuity, so any learned transition has large gradients.

The 2× wall weight difference means DeepFlow favors `u=0` more strongly near the corner, creating a sharper transition. Combined with random (vs Hammersley) interior sampling and different initialization, this leads to larger localized PDE residuals.

---

## 5. Recommended fixes for DeepFlow

1. **Combine wall MSE terms**. Merge top and bottom wall losses into a single MSE (like DeepXDE) to halve the wall weight and double the inlet-to-wall ratio from 5× to 10×.

2. **Use quasi-random interior sampling**. Replace random sampling with Hammersley or Latin Hypercube for better space-filling coverage near corners. DeepFlow already has `sampling_lhs` support.

3. **Smooth inlet profile**. Replace the step inlet with a ramp or parabolic profile over a small distance from the wall, removing the corner singularity.

4. **Glorot initialization**. Match DeepXDE's `"Glorot normal"` initialization for fairer comparison.

5. **Adaptive weighting / RAR sampling**. Use `df.calc_loss_weighted` or R3/RAR sampling to increase point density or loss weight in the high-residual corner region.

---

## 6. Takeaway

The previous diagnosis incorrectly claimed DeepFlow pools all BCs into one MSE. In reality, DeepFlow already uses per-geometry MSE terms. The actual differences are: (1) walls split vs combined (2× wall weight), (2) random vs Hammersley interior sampling, and (3) Kaiming vs Glorot initialization. These collectively explain why DeepXDE produces smoother corner behavior.
