from types import SimpleNamespace


def test_reference_solver_recomputes_after_pde_mutation(monkeypatch):
    import deepflow.reference.solver as solver_module
    from deepflow.pde import HeatEquation

    pde_area = SimpleNamespace(
        physics_type="PDE",
        PDE=HeatEquation(alpha=0.1),
        bound_list=[],
        negative_bound_list=[],
        range_t=(0.0, 1.0),
        ranges={0: (0.0, 1.0), 1: (0.0, 1.0)},
    )
    domain = SimpleNamespace(area_list=[pde_area], bound_list=[])

    class FakeAdapter:
        def __init__(self, boundary_resolution):
            pass

        def build(self, area, ngs, mesh_size):
            return SimpleNamespace(boundary_info={}, labels_by_bound={})

    solutions = []
    alphas = []

    def fake_solve_heat(self, *args):
        alphas.append(args[2].alpha)
        solution = object()
        solutions.append(solution)
        return solution

    monkeypatch.setattr(solver_module, "_load_ngsolve", lambda: object())
    monkeypatch.setattr(solver_module, "NetgenGeometryAdapter", FakeAdapter)
    monkeypatch.setattr(solver_module.ReferenceSolver, "_solve_heat", fake_solve_heat)

    solver = solver_module.ReferenceSolver()
    first = solver.solve(domain)
    pde_area.PDE.alpha = 0.2
    second = solver.solve(domain)

    assert first is solutions[0]
    assert second is solutions[1]
    assert first is not second
    assert alphas == [0.1, 0.2]

    # Keep the historical method callable for users that previously cleared it.
    solver.clear_cache()
