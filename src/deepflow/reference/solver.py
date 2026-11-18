"""Lazy-loaded NGSolve reference solver implementations."""

import math
import numbers
import warnings
from typing import Mapping, Optional, Tuple

import numpy as np

from ..geometry import Area, Bound
from ..nn import HardConstraint
from ..pde import (
    BurgersEquation1D,
    CustomPDE,
    HeatEquation,
    NavierStokes,
    StreamFunctionNavierStokes,
    WaveEquation,
)
from .geometry import (
    MeshGeometry,
    NetgenGeometryAdapter,
    ReferenceGeometryError,
    _sample_bound,
)
from .solution import ReferenceSolution


class ReferenceConfigurationError(ValueError):
    """Raised for a valid DeepFlow object that is unsupported in v1."""


class UnsupportedReferencePDE(NotImplementedError):
    """Raised when a PDE has no reference-backend implementation."""


def _load_ngsolve():
    try:
        import ngsolve
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The optional DeepFlow reference backend requires NGSolve/Netgen. "
            "Install it with `pip install deepflow[cfd]` in a supported "
            "Python 3.10+ environment."
        ) from exc
    return ngsolve


def _all_geometries(domain):
    return list(getattr(domain, "area_list", ())) + list(
        getattr(domain, "bound_list", ())
    )


def _time_interval(domain, pde_area) -> Optional[Tuple[float, float]]:
    candidates = [pde_area] + [item for item in _all_geometries(domain) if item is not pde_area]
    for item in candidates:
        value = getattr(item, "range_t", None)
        if value is None:
            continue
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
            raise ReferenceConfigurationError(
                "Transient reference problems require a time interval ``(t0, t1)``."
            )
        start, end = float(value[0]), float(value[1])
        if not np.isfinite([start, end]).all() or end <= start:
            raise ReferenceConfigurationError(
                "The transient time interval must satisfy finite t1 > t0."
            )
        return start, end
    return None


def _time_values(interval, time_step):
    start, end = interval
    if time_step is None:
        time_step = (end - start) / 100.0
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be a positive finite number")
    count = max(1, int(math.ceil((end - start) / time_step)))
    return np.linspace(start, end, count + 1)


def _default_mesh_size(area: Area) -> float:
    widths = [abs(float(area.ranges[axis][1] - area.ranges[axis][0])) for axis in (0, 1)]
    positive = [width for width in widths if width > 0]
    if not positive:
        raise ReferenceConfigurationError("The PDE area must have positive extent.")
    return min(positive) / 20.0


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) or isinstance(value, np.number)


def _validate_condition(condition, key: str, *, allow_time_derivative: bool = False):
    if isinstance(condition, HardConstraint):
        raise ReferenceConfigurationError(
            "HardConstraint boundary/initial conditions are not supported by "
            "the NGSolve reference backend v1."
        )
    if "_" in key and not (allow_time_derivative and key == "u_t"):
        raise ReferenceConfigurationError(
            f"Derivative or typed condition '{key}' is not supported by the "
            "v1 reference backend; use numeric or ('x'/'y', callable) Dirichlet values."
        )
    if _is_number(condition):
        return
    if isinstance(condition, (tuple, list)) and len(condition) == 2:
        variable, function = condition
        if variable not in ("x", "y") or not callable(function):
            raise ReferenceConfigurationError(
                f"Condition for '{key}' must be a number or ('x'/'y', callable)."
            )
        return
    raise ReferenceConfigurationError(
        f"Condition for '{key}' must be a number or ('x'/'y', callable)."
    )


def _validate_geometry_conditions(domain, supported_fields, *, transient=False):
    for item in _all_geometries(domain):
        if getattr(item, "physics_type", None) == "BC" and isinstance(item, Bound):
            for key, condition in (item.condition_dict or {}).items():
                _validate_condition(condition, key)
                if key not in supported_fields:
                    raise ReferenceConfigurationError(
                        f"Boundary field '{key}' is not declared by the selected PDE."
                    )

    for item in _all_geometries(domain):
        if getattr(item, "physics_type", None) == "IC":
            for key, condition in (item.condition_dict or {}).items():
                if key not in supported_fields and not (transient and key == "u_t"):
                    raise ReferenceConfigurationError(
                        f"Initial-condition field '{key}' is not supported by the selected PDE."
                    )
                _validate_condition(
                    condition, key, allow_time_derivative=transient and key == "u_t"
                )


def _polyline_distance(first: np.ndarray, second: np.ndarray) -> float:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(
        max(
            np.min(distances, axis=1).max(),
            np.min(distances, axis=0).max(),
        )
    )


def _is_point_like(bound: Bound, pde_area: Area) -> bool:
    scale = max(
        1.0,
        abs(float(pde_area.ranges[0][1] - pde_area.ranges[0][0])),
        abs(float(pde_area.ranges[1][1] - pde_area.ranges[1][0])),
    )
    return max(bound.lengths.values()) <= scale * 1.0e-5


def _boundary_label(bound: Bound, adapter: MeshGeometry):
    label = adapter.labels_by_bound.get(id(bound))
    if label is not None:
        return label

    matches = []
    for candidate_label, candidate_points in adapter.points_by_label.items():
        original_coordinates = (
            bound.X,
            bound.Y,
            dict(getattr(bound, "coords", {})),
        )
        try:
            points = _sample_bound(bound, len(candidate_points))
        finally:
            bound.X, bound.Y = original_coordinates[:2]
            bound.coords = original_coordinates[2]
        scale = max(
            1.0,
            float(np.ptp(candidate_points[:, 0])),
            float(np.ptp(candidate_points[:, 1])),
        )
        if _polyline_distance(points, candidate_points) <= scale * 1.0e-7:
            matches.append(candidate_label)

    if len(matches) > 1:
        raise ReferenceConfigurationError(
            "A standalone boundary matches multiple PDE-area boundaries; "
            "attach its BC to the Area boundary object explicitly."
        )
    return matches[0] if matches else None


def _collect_boundary_conditions(domain, pde_area, adapter: MeshGeometry):
    conditions = {label: {} for label in adapter.boundary_info}
    bounds = list(pde_area.bound_list) + list(pde_area.negative_bound_list or [])
    bounds += list(getattr(domain, "bound_list", ()))
    seen = set()
    for bound in bounds:
        if id(bound) in seen:
            continue
        seen.add(id(bound))
        if getattr(bound, "physics_type", None) == "BC":
            label = _boundary_label(bound, adapter)
            if label is None:
                if _is_point_like(bound, pde_area):
                    warnings.warn(
                        "Ignoring a point-like FEM boundary condition; "
                        "point Dirichlet constraints are not represented by "
                        "the NGSolve boundary labels.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                raise ReferenceConfigurationError(
                    "A boundary condition is attached to a Bound that is not "
                    "part of the PDE area boundary."
                )
            conditions[label].update(bound.condition_dict or {})
    return conditions


def _field_labels(boundary_conditions: Mapping[str, Mapping[str, object]], fields):
    labels = {field: [] for field in fields}
    for label, conditions in boundary_conditions.items():
        for field in fields:
            if field in conditions:
                labels[field].append(label)
    return labels


def _numeric_function_value(function, value):
    attempts = [value, np.asarray(value), np.asarray(value, dtype=float)]
    try:
        import torch

        attempts.append(torch.as_tensor(value, dtype=torch.get_default_dtype()))
    except ImportError:
        pass
    error = None
    for argument in attempts:
        try:
            result = function(argument)
            if hasattr(result, "detach"):
                result = result.detach().cpu().numpy()
            result = np.asarray(result)
            if result.size != 1:
                raise ValueError("boundary callable must return a scalar")
            return float(result.reshape(-1)[0])
        except Exception as exc:
            error = exc
    raise ValueError("Could not evaluate boundary callable at a mesh point") from error


def _polynomial_condition_cf(ngs, function, symbol, variable, coordinate_ranges=None):
    if coordinate_ranges is None:
        lower, upper = -1.0, 1.0
    else:
        axis = 0 if variable == "x" else 1
        lower, upper = coordinate_ranges[axis]
        lower, upper = float(lower), float(upper)
    if abs(upper - lower) <= 1.0e-14:
        return ngs.CoefficientFunction(_numeric_function_value(function, lower))
    samples = np.linspace(lower, upper, 33)
    values = np.asarray([_numeric_function_value(function, value) for value in samples])
    polynomial = np.polynomial.Polynomial.fit(samples, values, 16).convert()
    expression = sum(float(coefficient) * symbol ** index for index, coefficient in enumerate(polynomial.coef))
    return ngs.CoefficientFunction(expression)


def _condition_to_cf(ngs, condition, x_symbol, y_symbol, coordinate_ranges=None):
    if _is_number(condition):
        return ngs.CoefficientFunction(float(condition))
    variable, function = condition
    symbol = x_symbol if variable == "x" else y_symbol
    try:
        symbolic_value = function(symbol)
        return ngs.CoefficientFunction(symbolic_value)
    except Exception:
        return _polynomial_condition_cf(
            ngs, function, symbol, variable, coordinate_ranges
        )


def _set_boundary_values(ngs, geometry, target, boundary_conditions, field):
    vector = target.vec.FV().NumPy()
    for label, conditions in boundary_conditions.items():
        if field not in conditions:
            continue
        region = geometry.mesh.Boundaries(label)
        coefficient = _condition_to_cf(
            ngs, conditions[field], ngs.x, ngs.y, geometry.area.ranges
        )
        temporary = ngs.GridFunction(target.space)
        temporary.Set(coefficient, ngs.BND, definedon=region)
        dofs = target.space.GetDofs(region)
        boundary_vector = temporary.vec.FV().NumPy()
        for dof in range(vector.size):
            if dofs[dof]:
                vector[dof] = boundary_vector[dof]


def _set_initial_field(
    ngs,
    grid_function,
    condition,
    *,
    mesh=None,
    coordinate_ranges=None,
):
    if mesh is None:
        raise ValueError("An NGSolve mesh is required for initial field projection.")
    coefficient = _condition_to_cf(
        ngs, condition, ngs.x, ngs.y, coordinate_ranges
    )
    grid_function.Set(coefficient)


def _copy_grid_function(ngs, grid_function, fes):
    copied = ngs.GridFunction(fes)
    copied.vec.data = grid_function.vec
    return copied


def _relative_change(current, previous):
    difference = (current.vec - previous.vec).Norm()
    denominator = max(current.vec.Norm(), 1.0)
    return float(difference / denominator)


def _linear_solve(bilinear_form, linear_form, grid_function, fes):
    right_hand_side = linear_form.vec - bilinear_form.mat * grid_function.vec
    grid_function.vec.data += bilinear_form.mat.Inverse(fes.FreeDofs()) * right_hand_side


def _normalize_pressure(ngs, mesh, pressure):
    measure = ngs.Integrate(1, mesh)
    if abs(float(measure)) <= 1.0e-30:
        return 0.0
    mean = float(ngs.Integrate(pressure * ngs.dx, mesh) / measure)
    pressure.vec.FV().NumPy()[:] -= mean
    correction = float(ngs.Integrate(pressure * ngs.dx, mesh) / measure)
    if abs(correction) > 1.0e-13:
        pressure.vec.FV().NumPy()[:] -= correction
        mean += correction
    return mean


def _make_scalar_fes(ngs, mesh, labels, order=2):
    kwargs = {"order": order}
    if labels:
        kwargs["dirichlet"] = "|".join(labels)
    return ngs.H1(mesh, **kwargs)


def _make_navier_fes(ngs, mesh, labels):
    return ngs.FESpace([
        _make_scalar_fes(ngs, mesh, labels[field], order=order)
        for field, order in (("u", 2), ("v", 2), ("p", 1))
    ])


def _initial_conditions(domain):
    initial = {}
    for item in _all_geometries(domain):
        if getattr(item, "physics_type", None) == "IC":
            for key, condition in (item.condition_dict or {}).items():
                initial.setdefault(key, condition)
    return initial


class ReferenceSolver:
    """Solve built-in PDEs with optional NGSolve/Netgen FEM.

    Args:
        mesh_size: Target mesh size; ``None`` infers it from the area.
        boundary_resolution: Samples per mesh boundary.
        time_step: Transient snapshot spacing.
        tolerance: Solver convergence tolerance.
        max_iterations: Solver iteration limit.
    """

    def __init__(
        self,
        mesh_size=None,
        boundary_resolution=128,
        time_step=None,
        tolerance=1e-8,
        max_iterations=200,
    ):
        """Validate and store reference-solver configuration."""
        if mesh_size is not None and (not np.isfinite(mesh_size) or mesh_size <= 0):
            raise ValueError("mesh_size must be positive")
        if not isinstance(boundary_resolution, int) or boundary_resolution < 4:
            raise ValueError("boundary_resolution must be an integer >= 4")
        if time_step is not None and (not np.isfinite(time_step) or time_step <= 0):
            raise ValueError("time_step must be positive")
        if tolerance <= 0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be positive and finite")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.mesh_size = None if mesh_size is None else float(mesh_size)
        self.boundary_resolution = boundary_resolution
        self.time_step = None if time_step is None else float(time_step)
        self.tolerance = float(tolerance)
        self.max_iterations = max_iterations

    def clear_cache(self):
        """Compatibility no-op; the solver does not cache solutions."""

    def solve(self, domain) -> ReferenceSolution:
        """Return a ``ReferenceSolution`` for a domain with exactly one PDE area.

        Raises:
            ImportError: NGSolve/Netgen is unavailable.
            ReferenceConfigurationError: Unsupported geometry, conditions, or
                time settings.
            UnsupportedReferencePDE: No FEM backend for the PDE.
        """
        areas = list(getattr(domain, "area_list", ()))
        pde_areas = [area for area in areas if getattr(area, "physics_type", None) == "PDE"]
        if len(pde_areas) != 1:
            raise ReferenceConfigurationError(
                "ReferenceSolver requires exactly one Area with a defined PDE."
            )
        pde_area = pde_areas[0]
        pde = pde_area.PDE

        if isinstance(pde, CustomPDE):
            raise UnsupportedReferencePDE(
                "CustomPDE is not supported by the NGSolve reference backend."
            )
        supported = {"u"}
        if isinstance(pde, (NavierStokes, StreamFunctionNavierStokes)):
            supported = {"u", "v", "p"}
            if isinstance(pde, StreamFunctionNavierStokes):
                supported.add("psi")
        _validate_geometry_conditions(
            domain, supported,
            transient=isinstance(pde, (NavierStokes, HeatEquation, WaveEquation)),
        )

        interval = (
            None
            if isinstance(pde, BurgersEquation1D)
            else _time_interval(domain, pde_area)
        )
        if isinstance(pde, StreamFunctionNavierStokes) and interval is not None:
            raise ReferenceConfigurationError(
                "StreamFunctionNavierStokes supports steady reference problems only."
            )
        if isinstance(pde, (HeatEquation, WaveEquation)) and interval is None:
            raise ReferenceConfigurationError(
                f"{pde.__class__.__name__} reference problems require a time interval."
            )

        ngs = _load_ngsolve()
        mesh_size = self.mesh_size or _default_mesh_size(pde_area)
        adapter = NetgenGeometryAdapter(self.boundary_resolution)
        geometry = adapter.build(pde_area, ngs, mesh_size)
        boundary_conditions = _collect_boundary_conditions(domain, pde_area, geometry)
        if isinstance(pde, BurgersEquation1D):
            return self._solve_burgers(
                pde_area, pde, geometry, boundary_conditions, ngs
            )
        if isinstance(pde, StreamFunctionNavierStokes):
            return self._solve_stream_function(
                domain, pde_area, pde, geometry, boundary_conditions, ngs
            )
        if isinstance(pde, NavierStokes):
            return self._solve_navier_stokes(
                domain, pde_area, pde, geometry, boundary_conditions, interval, ngs
            )
        if isinstance(pde, HeatEquation):
            return self._solve_heat(
                domain, pde_area, pde, geometry, boundary_conditions, interval, ngs
            )
        if isinstance(pde, WaveEquation):
            return self._solve_wave(
                domain, pde_area, pde, geometry, boundary_conditions, interval, ngs
            )
        raise UnsupportedReferencePDE(
            f"Unsupported reference PDE: {pde.__class__.__name__}."
        )

    def _solution(self, pde, geometry, *, fields=None, snapshots=None, times=None, **details):
        field_names = fields if snapshots is None else snapshots[0]
        metadata = {
            "backend": "NGSolve/Netgen",
            "pde": pde.__class__.__name__,
            "fields": list(field_names),
            "mesh_size": float(self.mesh_size or _default_mesh_size(geometry.area)),
            "mesh": {
                "dimension": int(geometry.mesh.dim),
                "elements": int(geometry.mesh.ne),
                "vertices": int(geometry.mesh.nv),
            },
            "boundary_labels": geometry.boundary_info,
        }
        if times is not None:
            metadata.update(
                time_values=times.tolist(),
                time_step=float(np.min(np.diff(times))),
            )
        metadata.update(details)
        return ReferenceSolution(
            area=geometry.area,
            mesh=geometry.mesh,
            fields=fields,
            snapshots=snapshots,
            times=times,
            metadata=metadata,
        )

    def _solve_scalar_initial(
        self, ngs, fes, condition, *, mesh=None, coordinate_ranges=None
    ):
        field = ngs.GridFunction(fes)
        _set_initial_field(
            ngs, field, condition, mesh=mesh,
            coordinate_ranges=coordinate_ranges,
        )
        return field

    def _solve_heat(self, domain, pde_area, pde, geometry, boundary_conditions, interval, ngs):
        labels = _field_labels(boundary_conditions, ["u"])
        fes = _make_scalar_fes(ngs, geometry.mesh, labels["u"])
        initial = _initial_conditions(domain)
        if "u" not in initial:
            raise ReferenceConfigurationError(
                "HeatEquation reference solutions require an initial condition for u."
            )
        old = self._solve_scalar_initial(
            ngs, fes, initial["u"], mesh=geometry.mesh,
            coordinate_ranges=pde_area.ranges,
        )
        times = _time_values(interval, self.time_step)
        snapshots = [{"u": old}]
        residuals = []
        for previous_time, current_time in zip(times[:-1], times[1:]):
            dt = float(current_time - previous_time)
            trial = fes.TrialFunction()
            test = fes.TestFunction()
            bilinear = ngs.BilinearForm(fes)
            bilinear += (trial * test / dt + pde.alpha * ngs.grad(trial) * ngs.grad(test)) * ngs.dx
            linear = ngs.LinearForm(fes)
            linear += old * test / dt * ngs.dx
            bilinear.Assemble()
            linear.Assemble()
            current = ngs.GridFunction(fes)
            current.vec.data = old.vec
            _set_boundary_values(
                ngs, geometry, current, boundary_conditions, "u"
            )
            _linear_solve(bilinear, linear, current, fes)
            residuals.append(_relative_change(current, old))
            snapshots.append({"u": current})
            old = current

        return self._solution(
            pde,
            geometry,
            snapshots=snapshots,
            times=times,
            iterations=[1] * (len(times) - 1),
            solver_residuals=residuals,
            converged=True,
        )

    def _solve_wave(self, domain, pde_area, pde, geometry, boundary_conditions, interval, ngs):
        labels = _field_labels(boundary_conditions, ["u"])
        fes = _make_scalar_fes(ngs, geometry.mesh, labels["u"])
        initial = _initial_conditions(domain)
        if "u" not in initial:
            raise ReferenceConfigurationError(
                "WaveEquation reference solutions require an initial displacement u."
            )
        displacement = self._solve_scalar_initial(
            ngs, fes, initial["u"], mesh=geometry.mesh,
            coordinate_ranges=pde_area.ranges,
        )
        velocity = ngs.GridFunction(fes)
        velocity.Set(0)
        if "u_t" in initial:
            _set_initial_field(
                ngs, velocity, initial["u_t"], mesh=geometry.mesh,
                coordinate_ranges=pde_area.ranges,
            )
        acceleration = ngs.GridFunction(fes)
        acceleration.Set(0)
        times = _time_values(interval, self.time_step)
        snapshots = [{"u": displacement}]
        residuals = []

        mass = ngs.BilinearForm(fes)
        mass_trial = fes.TrialFunction()
        mass_test = fes.TestFunction()
        mass += mass_trial * mass_test * ngs.dx
        mass.Assemble()
        acceleration_rhs = ngs.LinearForm(fes)
        acceleration_rhs += -pde.c**2 * ngs.grad(displacement) * ngs.grad(mass_test) * ngs.dx
        acceleration_rhs.Assemble()
        _linear_solve(mass, acceleration_rhs, acceleration, fes)

        beta, gamma = 0.25, 0.5
        for previous_time, current_time in zip(times[:-1], times[1:]):
            dt = float(current_time - previous_time)
            trial = fes.TrialFunction()
            test = fes.TestFunction()
            predictor = (
                displacement
                + dt * velocity
                + dt * dt * (0.5 - beta) * acceleration
            )
            bilinear = ngs.BilinearForm(fes)
            bilinear += (
                trial * test / (beta * dt * dt)
                + pde.c**2 * ngs.grad(trial) * ngs.grad(test)
            ) * ngs.dx
            linear = ngs.LinearForm(fes)
            linear += predictor * test / (beta * dt * dt) * ngs.dx
            bilinear.Assemble()
            linear.Assemble()
            new_displacement = ngs.GridFunction(fes)
            new_displacement.vec.data = displacement.vec
            _set_boundary_values(
                ngs, geometry, new_displacement, boundary_conditions, "u"
            )
            _linear_solve(bilinear, linear, new_displacement, fes)

            new_acceleration = ngs.GridFunction(fes)
            new_acceleration.Set(0)
            acceleration_rhs = ngs.LinearForm(fes)
            acceleration_rhs += -pde.c**2 * ngs.grad(new_displacement) * ngs.grad(mass_test) * ngs.dx
            acceleration_rhs.Assemble()
            _linear_solve(mass, acceleration_rhs, new_acceleration, fes)
            new_velocity = ngs.GridFunction(fes)
            new_velocity.vec.data = velocity.vec + dt * (
                (1.0 - gamma) * acceleration.vec + gamma * new_acceleration.vec
            )
            residuals.append(_relative_change(new_displacement, displacement))
            snapshots.append({"u": new_displacement})
            displacement, velocity, acceleration = (
                new_displacement,
                new_velocity,
                new_acceleration,
            )

        return self._solution(
            pde,
            geometry,
            snapshots=snapshots,
            times=times,
            iterations=[1] * (len(times) - 1),
            solver_residuals=residuals,
            converged=True,
            time_integrator="implicit Newmark (beta=0.25, gamma=0.5)",
        )

    def _navier_bilinear(self, ngs, fes, pde, previous, dt, pressure_gauge):
        trial_u, trial_v, trial_p = fes.TrialFunction()
        test_u, test_v, test_p = fes.TestFunction()
        bilinear = ngs.BilinearForm(fes, symmetric=False)
        if dt is not None:
            bilinear += pde.rho / dt * (trial_u * test_u + trial_v * test_v) * ngs.dx
        bilinear += pde.mu * (
            ngs.grad(trial_u) * ngs.grad(test_u)
            + ngs.grad(trial_v) * ngs.grad(test_v)
        ) * ngs.dx
        previous_u, previous_v, _ = previous.components
        bilinear += pde.rho * (
            (previous_u * ngs.grad(trial_u)[0] + previous_v * ngs.grad(trial_u)[1]) * test_u
            + (previous_u * ngs.grad(trial_v)[0] + previous_v * ngs.grad(trial_v)[1]) * test_v
        ) * ngs.dx
        bilinear += -trial_p * (ngs.grad(test_u)[0] + ngs.grad(test_v)[1]) * ngs.dx
        bilinear += -test_p * (ngs.grad(trial_u)[0] + ngs.grad(trial_v)[1]) * ngs.dx
        if pressure_gauge:
            bilinear += 1.0e-12 * trial_p * test_p * ngs.dx
        return bilinear

    def _solve_navier_state(
        self,
        ngs,
        geometry,
        pde,
        fes,
        boundary_conditions,
        old_state=None,
        dt=None,
        pressure_gauge=False,
    ):
        current = ngs.GridFunction(fes)
        if old_state is not None:
            current.vec.data = old_state.vec
        for component, field in enumerate(("u", "v", "p")):
            _set_boundary_values(
                ngs, geometry, current.components[component], boundary_conditions, field
            )
        residuals = []
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            previous = _copy_grid_function(ngs, current, fes)
            bilinear = self._navier_bilinear(
                ngs, fes, pde, previous, dt, pressure_gauge
            )
            linear = ngs.LinearForm(fes)
            if dt is not None and old_state is not None:
                old_u, old_v, _ = old_state.components
                test_u, test_v, _ = fes.TestFunction()
                linear += pde.rho / dt * (
                    old_u * test_u + old_v * test_v
                ) * ngs.dx
            bilinear.Assemble()
            linear.Assemble()
            _linear_solve(bilinear, linear, current, fes)
            if pressure_gauge:
                _normalize_pressure(ngs, geometry.mesh, current.components[2])
            change = _relative_change(current, previous)
            residuals.append(change)
            if change <= self.tolerance:
                converged = True
                break
        return current, residuals, iteration, converged

    def _solve_navier_stokes(
        self, domain, pde_area, pde, geometry, boundary_conditions, interval, ngs
    ):
        labels = _field_labels(boundary_conditions, ["u", "v", "p"])
        fes = _make_navier_fes(ngs, geometry.mesh, labels)
        pressure_gauge = not bool(labels["p"])
        initial = _initial_conditions(domain)
        if interval is not None and not {"u", "v"}.issubset(initial):
            raise ReferenceConfigurationError(
                "Transient NavierStokes reference solutions require initial u and v values."
            )

        if interval is None:
            current, residuals, iteration_count, converged = self._solve_navier_state(
                ngs,
                geometry,
                pde,
                fes,
                boundary_conditions,
                pressure_gauge=pressure_gauge,
            )
            return self._solution(
                pde,
                geometry,
                fields={
                    "u": current.components[0],
                    "v": current.components[1],
                    "p": current.components[2],
                },
                iterations=iteration_count,
                solver_residuals=residuals,
                converged=converged,
                pressure_gauge=("zero_mean_stabilization" if pressure_gauge else "configured Dirichlet"),
            )

        initial_state = ngs.GridFunction(fes)
        for component, field in zip(initial_state.components, ("u", "v", "p")):
            if field in initial:
                _set_initial_field(
                    ngs, component, initial[field], mesh=geometry.mesh,
                    coordinate_ranges=pde_area.ranges,
                )
        times = _time_values(interval, self.time_step)
        snapshots = [
            {
                "u": initial_state.components[0],
                "v": initial_state.components[1],
                "p": initial_state.components[2],
            }
        ]
        iteration_counts, solver_residuals = [], []
        converged = True
        old_state = initial_state
        for previous_time, current_time in zip(times[:-1], times[1:]):
            dt = float(current_time - previous_time)
            current, residuals, iteration_count, step_converged = self._solve_navier_state(
                ngs,
                geometry,
                pde,
                fes,
                boundary_conditions,
                old_state=old_state,
                dt=dt,
                pressure_gauge=pressure_gauge,
            )
            old_state = current
            snapshots.append(
                {"u": current.components[0], "v": current.components[1], "p": current.components[2]}
            )
            iteration_counts.append(iteration_count)
            solver_residuals.append(residuals[-1] if residuals else math.inf)
            converged = converged and step_converged

        return self._solution(
            pde,
            geometry,
            snapshots=snapshots,
            times=times,
            iterations=iteration_counts,
            solver_residuals=solver_residuals,
            converged=converged,
            pressure_gauge=("zero_mean_stabilization" if pressure_gauge else "configured Dirichlet"),
        )

    def _solve_stream_function(
        self, domain, pde_area, pde, geometry, boundary_conditions, ngs
    ):
        navier_solution = self._solve_navier_stokes(
            domain, pde_area, NavierStokes(pde.mu, pde.rho, pde.U, pde.L), geometry,
            boundary_conditions, None, ngs
        )
        psi_conditions = {
            label: conditions for label, conditions in boundary_conditions.items() if "psi" in conditions
        }
        psi_labels = list(psi_conditions)
        if not psi_labels:
            psi_labels = [next(iter(geometry.boundary_info))]
            psi_conditions = {label: {"psi": 0.0} for label in psi_labels}
        psi_fes = _make_scalar_fes(ngs, geometry.mesh, psi_labels)
        psi = ngs.GridFunction(psi_fes)
        _set_boundary_values(
            ngs, geometry, psi, psi_conditions, "psi"
        )
        u = navier_solution._fields["u"]
        v = navier_solution._fields["v"]
        trial = psi_fes.TrialFunction()
        test = psi_fes.TestFunction()
        bilinear = ngs.BilinearForm(psi_fes)
        bilinear += ngs.grad(trial) * ngs.grad(test) * ngs.dx
        linear = ngs.LinearForm(psi_fes)
        linear += (-v * ngs.grad(test)[0] + u * ngs.grad(test)[1]) * ngs.dx
        bilinear.Assemble()
        linear.Assemble()
        _linear_solve(bilinear, linear, psi, psi_fes)

        metadata = dict(navier_solution.metadata)
        metadata["fields"] = ["psi", "u", "v", "p"]
        metadata["derived_fields"] = ["psi"]
        metadata["stream_function"] = "psi reconstructed by an H1 gradient-projection solve"
        return ReferenceSolution(
            area=pde_area,
            mesh=geometry.mesh,
            fields={"psi": psi, "u": u, "v": v, "p": navier_solution._fields["p"]},
            metadata=metadata,
        )

    def _solve_burgers(
        self, pde_area, pde, geometry, boundary_conditions, ngs
    ):
        labels = _field_labels(boundary_conditions, ["u"])
        fes = _make_scalar_fes(ngs, geometry.mesh, labels["u"])
        current = ngs.GridFunction(fes)
        _set_boundary_values(
            ngs, geometry, current, boundary_conditions, "u"
        )

        residuals = []
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            previous = _copy_grid_function(ngs, current, fes)
            trial = fes.TrialFunction()
            test = fes.TestFunction()
            bilinear = ngs.BilinearForm(fes, symmetric=False)
            bilinear += (
                ngs.grad(trial)[1] * test
                + previous * ngs.grad(trial)[0] * test
                + pde.nu * ngs.grad(trial)[0] * ngs.grad(test)[0]
            ) * ngs.dx
            linear = ngs.LinearForm(fes)
            bilinear.Assemble()
            linear.Assemble()
            _linear_solve(bilinear, linear, current, fes)
            _set_boundary_values(
                ngs, geometry, current, boundary_conditions, "u"
            )
            change = _relative_change(current, previous)
            residuals.append(change)
            if change <= self.tolerance:
                converged = True
                break

        return self._solution(
            pde,
            geometry,
            fields={"u": current},
            spatial_coordinates=["x", "y"],
            iterations=iteration,
            solver_residuals=residuals,
            converged=converged,
        )
__all__ = [
    "ReferenceConfigurationError",
    "ReferenceGeometryError",
    "ReferenceSolver",
    "UnsupportedReferencePDE",
]
