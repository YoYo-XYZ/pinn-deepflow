# Confirmed bugs

Seven defects are reproduced by 12 strict `xfail` cases. These tests assert intended behavior. Remove the relevant markers when fixing each defect.

| Affected behavior | Confirmed defect | Regression tests |
| --- | --- | --- |
| PDE differentiation | Constant fields and linear fields requiring second derivatives can raise `RuntimeError`, preventing evaluation of valid solutions. | [test_pde.py](../../../tests/test_pde.py) |
| RAR sampling | Original samples are lost; the result contains candidates plus selected points instead of original samples plus the requested additions. | [test_domain.py](../../../tests/test_domain.py) |
| R3 sampling around holes | Rejected replacement candidates leave fewer points than the requested fixed budget. | [test_domain.py](../../../tests/test_domain.py) |
| Coordinate restoration | `load_coordinates()` restores spatial coordinates but leaves processed model inputs unchanged. Subsequent losses use the later samples. | [test_domain.py](../../../tests/test_domain.py) |
| Fixed-time initial conditions | Supplying a scalar time to an IC causes `TypeError` during coordinate processing instead of producing a fixed time slice. | [test_physics.py](../../../tests/test_physics.py) |
| Evaluation at zero time | Predictions can be evaluated at `t=0`, but exported data omits the `t` field. | [test_evaluation.py](../../../tests/test_evaluation.py) |
| Direct area construction | `Area(boundaries).contains()` can accept exterior points outside both coordinate ranges. | [test_geometry.py](../../../tests/test_geometry.py) |