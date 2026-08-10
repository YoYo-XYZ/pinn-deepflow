"""Lazy-loaded NGSolve reference solver implementations."""

import math
import numbers
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
from .geometry import MeshGeometry, NetgenGeometryAdapter, ReferenceGeometryError
from .solution import ReferenceSolution


class ReferenceConfigurationError(ValueError):
    """Raised for a valid DeepFlow object that is unsupported in v1."""


class UnsupportedReferencePDE(NotImplementedError):
    """Raised when a PDE has no reference-backend implementation."""


def _load_ngsolve():
    """Import the optional backend only from the solve path."""
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


def _validate_geometry_conditions(domain, pde_area, supported_fields, *, transient=False):
    boundary_ids = {id(bound) for bound in pde_area.bound_list}
    boundary_ids.update(id(bound) for bound in (pde_area.negative_bound_list or []))
    for item in _all_geometries(domain):
        if getattr(item, "physics_type", None) == "BC" and (
            isinstance(item, Bound) and id(item) in boundary_ids
        ):
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


def _collect_boundary_conditions(domain, pde_area, adapter: MeshGeometry):
    conditions = {label: {} for label in adapter.boundary_info}
    for bound in list(pde_area.bound_list) + list(pde_area.negative_bound_list or []):
        label = adapter.labels_by_bound.get(id(bound))
        if label is None:
            continue
        if getattr(bound, "physics_type", None) == "BC":
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
    """Evaluate a DeepFlow callable with Python, NumPy, or Torch scalars."""
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
        except Exception as exc:  # try the next scalar representation
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
    degree = min(16, len(samples) - 1)
    polynomial = np.polynomial.Polynomial.fit(samples, values, degree).convert()
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


def _condition_to_cf_1d(ngs, condition, x_symbol, x_range=None):
    if _is_number(condition):
        return ngs.CoefficientFunction(float(condition))
    variable, function = condition
    if variable == "x":
        try:
            return ngs.CoefficientFunction(function(x_symbol))
        except Exception:
            ranges = {0: x_range or (-1.0, 1.0), 1: (0.0, 0.0)}
            return _polynomial_condition_cf(ngs, function, x_symbol, "x", ranges)
    return ngs.CoefficientFunction(_numeric_function_value(function, 0.0))


def _set_boundary_values(
    ngs,
    mesh,
    grid_function,
    boundary_conditions,
    field,
    *,
    component=None,
    points_by_label=None,
    coordinate_ranges=None,
):
    target = grid_function if component is None else grid_function.components[component]
    x_symbol = getattr(ngs, "x", None)
    y_symbol = getattr(ngs, "y", None)
    selected = {
        label: conditions[field]
        for label, conditions in boundary_conditions.items()
        if field in conditions
    }
    if not selected:
        return
    if points_by_label is None:
        for label, condition in selected.items():
            coefficient = _condition_to_cf(
                ngs, condition, x_symbol, y_symbol, coordinate_ranges
            )
            target.Set(coefficient, ngs.BND, definedon=mesh.Boundaries(label))
        return

    vector = target.vec.FV().NumPy()
    for label, condition in selected.items():
        temporary = ngs.GridFunction(target.space)
        coefficient = _condition_to_cf(
            ngs, condition, x_symbol, y_symbol, coordinate_ranges
        )
        temporary.Set(coefficient, ngs.BND, definedon=mesh.Boundaries(label))
        dofs = target.space.GetDofs(mesh.Boundaries(label))
        for dof in range(vector.size):
            if dofs[dof]:
                vector[dof] = temporary.vec.FV().NumPy()[dof]


def _assign_gridfunction_condition(ngs, grid_function, condition, coordinate_ranges):
    coefficient = _condition_to_cf(
        ngs, condition, ngs.x, ngs.y, coordinate_ranges
    )
    grid_function.Set(coefficient)


def _set_initial_field(
    ngs,
    grid_function,
    condition,
    *,
    one_dimensional=False,
    mesh=None,
    coordinate_ranges=None,
):
    if mesh is None:
        raise ValueError("An NGSolve mesh is required for initial field projection.")
    if one_dimensional:
        coefficient = _condition_to_cf_1d(
            ngs, condition, ngs.x,
            None if coordinate_ranges is None else coordinate_ranges[0],
        )
        grid_function.Set(coefficient)
    else:
        _assign_gridfunction_condition(
            ngs, grid_function, condition, coordinate_ranges
        )


def _copy_grid_function(ngs, grid_function, fes):
    copied = ngs.GridFunction(fes)
    copied.vec.data = grid_function.vec
    return copied


def _relative_change(current, previous):
    difference = (current.vec - previous.vec).Norm()
    denominator = max(current.vec.Norm(), 1.0)
    return float(difference / denominator)


def _linear_solve(ngs, bilinear_form, linear_form, grid_function, fes):
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
    velocity_kwargs = {"order": 2}
    if labels["u"]:
        velocity_kwargs["dirichlet"] = "|".join(labels["u"])
    u_space = ngs.H1(mesh, **velocity_kwargs)
    velocity_kwargs = {"order": 2}
    if labels["v"]:
        velocity_kwargs["dirichlet"] = "|".join(labels["v"])
    v_space = ngs.H1(mesh, **velocity_kwargs)
    pressure_kwargs = {"order": 1}
    if labels["p"]:
        pressure_kwargs["dirichlet"] = "|".join(labels["p"])
    pressure_space = ngs.H1(mesh, **pressure_kwargs)
    return ngs.FESpace([u_space, v_space, pressure_space])


def _initial_conditions(domain):
    initial = {}
    for item in _all_geometries(domain):
        if getattr(item, "physics_type", None) == "IC":
            for key, condition in (item.condition_dict or {}).items():
                initial.setdefault(key, condition)
    return initial


def _burgers_initial_from_boundary(domain, start_time):
    candidates = []
    for item in _all_geometries(domain):
        if not isinstance(item, Bound) or getattr(item, "physics_type", None) not in ("BC", "IC"):
            continue
        y_range = item.ranges.get(1)
        x_range = item.ranges.get(0)
        if y_range is None or x_range is None:
            continue
        if abs(float(y_range[0]) - start_time) <= 1.0e-7 * max(1.0, abs(start_time)) and abs(
            float(y_range[1]) - start_time
        ) <= 1.0e-7 * max(1.0, abs(start_time)):
            if float(x_range[1]) - float(x_range[0]) > 0 and "u" in (item.condition_dict or {}):
                candidates.append(item.condition_dict["u"])
    return candidates[0] if candidates else None


def _make_solution_evaluator(mesh):
    def evaluate(field, x, y):
        values = np.empty(x.size, dtype=float)
        for index, (x_value, y_value) in enumerate(zip(x.flat, y.flat)):
            values[index] = float(np.real(field(mesh(float(x_value), float(y_value)))))
        return values

    return evaluate


def _make_one_dimensional_evaluator(mesh):
    def evaluate(field, x, y):
        values = np.empty(x.size, dtype=float)
        for index, x_value in enumerate(x.flat):
            values[index] = float(np.real(field(mesh(float(x_value)))))
        return values

    return evaluate


class ReferenceSolver:
    """Solve DeepFlow's built-in PDEs with optional NGSolve/Netgen FEM."""

    def __init__(
        self,
        mesh_size=None,
        boundary_resolution=128,
        time_step=None,
        tolerance=1e-8,
        max_iterations=200,
    ):
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
        """Retain the historical cache API; solutions are not solver-cached."""

    def solve(self, domain) -> ReferenceSolution:
        """Solve the PDE attached to one ``ProblemDomain`` area."""
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
            domain,
            pde_area,
            supported,
            transient=isinstance(pde, (NavierStokes, HeatEquation, WaveEquation, BurgersEquation1D)),
        )

        interval = _time_interval(domain, pde_area)
        if isinstance(pde, StreamFunctionNavierStokes) and interval is not None:
            raise ReferenceConfigurationError(
                "StreamFunctionNavierStokes supports steady reference problems only."
            )
        if isinstance(pde, (HeatEquation, WaveEquation, BurgersEquation1D)) and interval is None:
            raise ReferenceConfigurationError(
                f"{pde.__class__.__name__} reference problems require a time interval."
            )

        ngs = _load_ngsolve()
        if isinstance(pde, BurgersEquation1D):
            solution = self._solve_burgers(domain, pde_area, pde, interval, ngs)
        else:
            mesh_size = self.mesh_size or _default_mesh_size(pde_area)
            adapter = NetgenGeometryAdapter(self.boundary_resolution)
            geometry = adapter.build(pde_area, ngs, mesh_size)
            boundary_conditions = _collect_boundary_conditions(domain, pde_area, geometry)
            if isinstance(pde, StreamFunctionNavierStokes):
                solution = self._solve_stream_function(
                    domain, pde_area, pde, geometry, boundary_conditions, ngs
                )
            elif isinstance(pde, NavierStokes):
                solution = self._solve_navier_stokes(
                    domain, pde_area, pde, geometry, boundary_conditions, interval, ngs
                )
            elif isinstance(pde, HeatEquation):
                solution = self._solve_heat(
                    domain, pde_area, pde, geometry, boundary_conditions, interval, ngs
                )
            elif isinstance(pde, WaveEquation):
                solution = self._solve_wave(
                    domain, pde_area, pde, geometry, boundary_conditions, interval, ngs
                )
            else:
                raise UnsupportedReferencePDE(
                    f"Unsupported reference PDE: {pde.__class__.__name__}."
                )

        return solution

    @staticmethod
    def _metadata(pde, geometry, mesh_size, field_names, **extra):
        metadata = {
            "backend": "NGSolve/Netgen",
            "pde": pde.__class__.__name__,
            "fields": list(field_names),
            "mesh_size": float(mesh_size),
            "mesh": {
                "dimension": int(geometry.mesh.dim),
                "elements": int(geometry.mesh.ne),
                "vertices": int(geometry.mesh.nv),
            },
            "boundary_labels": geometry.boundary_info,
        }
        metadata.update(extra)
        return metadata

    def _solve_scalar_initial(
        self, ngs, fes, condition, *, one_dimensional=False, mesh=None,
        coordinate_ranges=None
    ):
        field = ngs.GridFunction(fes)
        _set_initial_field(
            ngs, field, condition, one_dimensional=one_dimensional, mesh=mesh,
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
        iterations = []
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
                ngs, geometry.mesh, current, boundary_conditions, "u",
                points_by_label=geometry.points_by_label,
                coordinate_ranges=pde_area.ranges,
            )
            _linear_solve(ngs, bilinear, linear, current, fes)
            residuals.append(_relative_change(current, old))
            iterations.append(1)
            snapshots.append({"u": current})
            old = current

        # Scalar implicit steps are solved directly.  The recorded change is
        # a diagnostic of physical evolution, not a nonlinear convergence test.
        converged = True
        metadata = self._metadata(
            pde,
            geometry,
            self.mesh_size or _default_mesh_size(pde_area),
            ["u"],
            time_values=times.tolist(),
            time_step=float(np.min(np.diff(times))),
            iterations=iterations,
            solver_residuals=residuals,
            converged=converged,
        )
        return ReferenceSolution(
            area=pde_area,
            mesh=geometry.mesh,
            snapshots=snapshots,
            times=times,
            metadata=metadata,
            field_evaluator=_make_solution_evaluator(geometry.mesh),
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

        # Compute a consistent initial acceleration from M*a + K*u = 0.
        mass = ngs.BilinearForm(fes)
        mass_trial = fes.TrialFunction()
        mass_test = fes.TestFunction()
        mass += mass_trial * mass_test * ngs.dx
        mass.Assemble()
        acceleration_rhs = ngs.LinearForm(fes)
        acceleration_rhs += -pde.c**2 * ngs.grad(displacement) * ngs.grad(mass_test) * ngs.dx
        acceleration_rhs.Assemble()
        _linear_solve(ngs, mass, acceleration_rhs, acceleration, fes)

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
                ngs, geometry.mesh, new_displacement, boundary_conditions, "u",
                points_by_label=geometry.points_by_label,
                coordinate_ranges=pde_area.ranges,
            )
            _linear_solve(ngs, bilinear, linear, new_displacement, fes)

            new_acceleration = ngs.GridFunction(fes)
            new_acceleration.Set(0)
            acceleration_rhs = ngs.LinearForm(fes)
            acceleration_rhs += -pde.c**2 * ngs.grad(new_displacement) * ngs.grad(mass_test) * ngs.dx
            acceleration_rhs.Assemble()
            _linear_solve(ngs, mass, acceleration_rhs, new_acceleration, fes)
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

        metadata = self._metadata(
            pde,
            geometry,
            self.mesh_size or _default_mesh_size(pde_area),
            ["u"],
            time_values=times.tolist(),
            time_step=float(np.min(np.diff(times))),
            iterations=[1] * (len(times) - 1),
            solver_residuals=residuals,
            converged=True,
            time_integrator="implicit Newmark (beta=0.25, gamma=0.5)",
        )
        return ReferenceSolution(
            area=pde_area,
            mesh=geometry.mesh,
            snapshots=snapshots,
            times=times,
            metadata=metadata,
            field_evaluator=_make_solution_evaluator(geometry.mesh),
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
        initial_state=None,
        dt=None,
        pressure_gauge=False,
    ):
        current = ngs.GridFunction(fes)
        if initial_state is not None:
            current.vec.data = initial_state.vec
        elif old_state is not None:
            current.vec.data = old_state.vec
        for component, field in enumerate(("u", "v", "p")):
            _set_boundary_values(
                ngs,
                geometry.mesh,
                current,
                boundary_conditions,
                field,
                component=component,
                points_by_label=geometry.points_by_label,
                coordinate_ranges=geometry.area.ranges,
            )
        previous_time_state = old_state
        residuals = []
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            previous = _copy_grid_function(ngs, current, fes)
            bilinear = self._navier_bilinear(
                ngs, fes, pde, previous, dt, pressure_gauge
            )
            linear = ngs.LinearForm(fes)
            if dt is not None and previous_time_state is not None:
                old_u, old_v, _ = previous_time_state.components
                test_u, test_v, _ = fes.TestFunction()
                linear += pde.rho / dt * (
                    old_u * test_u + old_v * test_v
                ) * ngs.dx
            bilinear.Assemble()
            linear.Assemble()
            _linear_solve(ngs, bilinear, linear, current, fes)
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
        transient = interval is not None
        if transient and not {"u", "v"}.issubset(initial):
            raise ReferenceConfigurationError(
                "Transient NavierStokes reference solutions require initial u and v values."
            )

        if transient:
            initial_state = ngs.GridFunction(fes)
            _set_initial_field(
                ngs, initial_state.components[0], initial["u"], mesh=geometry.mesh,
                coordinate_ranges=pde_area.ranges,
            )
            _set_initial_field(
                ngs, initial_state.components[1], initial["v"], mesh=geometry.mesh,
                coordinate_ranges=pde_area.ranges,
            )
            if "p" in initial:
                _set_initial_field(
                    ngs, initial_state.components[2], initial["p"], mesh=geometry.mesh,
                    coordinate_ranges=pde_area.ranges,
                )
        else:
            initial_state = None

        if not transient:
            current, residuals, iteration_count, converged = self._solve_navier_state(
                ngs,
                geometry,
                pde,
                fes,
                boundary_conditions,
                initial_state=None,
                pressure_gauge=pressure_gauge,
            )
            metadata = self._metadata(
                pde,
                geometry,
                self.mesh_size or _default_mesh_size(pde_area),
                ["u", "v", "p"],
                iterations=iteration_count,
                solver_residuals=residuals,
                converged=converged,
                pressure_gauge=("zero_mean_stabilization" if pressure_gauge else "configured Dirichlet"),
            )
            return ReferenceSolution(
                area=pde_area,
                mesh=geometry.mesh,
                fields={
                    "u": current.components[0],
                    "v": current.components[1],
                    "p": current.components[2],
                },
                metadata=metadata,
                field_evaluator=_make_solution_evaluator(geometry.mesh),
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

        metadata = self._metadata(
            pde,
            geometry,
            self.mesh_size or _default_mesh_size(pde_area),
            ["u", "v", "p"],
            time_values=times.tolist(),
            time_step=float(np.min(np.diff(times))),
            iterations=iteration_counts,
            solver_residuals=solver_residuals,
            converged=converged,
            pressure_gauge=("zero_mean_stabilization" if pressure_gauge else "configured Dirichlet"),
        )
        return ReferenceSolution(
            area=pde_area,
            mesh=geometry.mesh,
            snapshots=snapshots,
            times=times,
            metadata=metadata,
            field_evaluator=_make_solution_evaluator(geometry.mesh),
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
            ngs, geometry.mesh, psi, psi_conditions, "psi",
            points_by_label=geometry.points_by_label,
            coordinate_ranges=pde_area.ranges,
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
        _linear_solve(ngs, bilinear, linear, psi, psi_fes)

        metadata = dict(navier_solution.metadata)
        metadata["fields"] = ["psi", "u", "v", "p"]
        metadata["derived_fields"] = ["psi"]
        metadata["stream_function"] = "psi reconstructed by an H1 gradient-projection solve"
        return ReferenceSolution(
            area=pde_area,
            mesh=geometry.mesh,
            fields={"psi": psi, "u": u, "v": v, "p": navier_solution._fields["p"]},
            metadata=metadata,
            field_evaluator=_make_solution_evaluator(geometry.mesh),
        )

    def _make_burgers_mesh(self, ngs, x_range, mesh_size):
        from netgen.meshing import Element0D, Element1D, FaceDescriptor, Mesh as NetgenMesh, MeshPoint, Pnt

        count = max(2, int(math.ceil((x_range[1] - x_range[0]) / mesh_size)))
        netgen_mesh = NetgenMesh(dim=1)
        descriptor = FaceDescriptor(surfnr=1, domin=1, domout=0, bc=0)
        descriptor.bcname = "default"
        netgen_mesh.Add(descriptor)
        points = [
            netgen_mesh.Add(MeshPoint(Pnt(float(value), 0.0, 0.0)))
            for value in np.linspace(x_range[0], x_range[1], count + 1)
        ]
        for first, second in zip(points[:-1], points[1:]):
            netgen_mesh.Add(Element1D([first, second], index=1))
        netgen_mesh.Add(Element0D(points[0], index=1))
        netgen_mesh.Add(Element0D(points[-1], index=1))
        netgen_mesh.AddRegion("spatial", 1)
        return ngs.Mesh(netgen_mesh)

    def _solve_burgers(self, domain, pde_area, pde, interval, ngs):
        x_range = tuple(float(value) for value in pde_area.ranges[0])
        mesh_size = self.mesh_size or (x_range[1] - x_range[0]) / 40.0
        mesh = self._make_burgers_mesh(ngs, x_range, mesh_size)
        initial = _initial_conditions(domain)
        if "u" not in initial:
            initial["u"] = _burgers_initial_from_boundary(domain, interval[0])
        if initial.get("u") is None:
            raise ReferenceConfigurationError(
                "BurgersEquation1D reference solutions require an initial u value "
                "on an IC geometry or the lower y boundary."
            )

        # The existing DeepFlow convention has x as space and y as time.  A
        # one-dimensional mesh is therefore sufficient for every time slice.
        boundary_values = {"left": None, "right": None}
        midpoint = 0.5 * (x_range[0] + x_range[1])
        for item in _all_geometries(domain):
            if not isinstance(item, Bound) or getattr(item, "physics_type", None) != "BC":
                continue
            x_bounds, y_bounds = item.ranges.get(0), item.ranges.get(1)
            if x_bounds is None or y_bounds is None or y_bounds[1] - y_bounds[0] <= 1.0e-8:
                continue
            if abs(x_bounds[0] - x_range[0]) <= 1.0e-7 and "u" in (item.condition_dict or {}):
                boundary_values["left"] = item.condition_dict["u"]
            if abs(x_bounds[1] - x_range[1]) <= 1.0e-7 and "u" in (item.condition_dict or {}):
                boundary_values["right"] = item.condition_dict["u"]

        labels = ["default"] if any(value is not None for value in boundary_values.values()) else []
        fes = _make_scalar_fes(ngs, mesh, labels)
        old = self._solve_scalar_initial(
            ngs, fes, initial["u"], one_dimensional=True, mesh=mesh,
            coordinate_ranges=(x_range, pde_area.ranges[1]),
        )

        def set_endpoint_values(grid_function):
            if not labels:
                return
            vector = grid_function.vec.FV().NumPy()
            for element in mesh.Elements(ngs.BND):
                dofs = fes.GetDofNrs(element)
                for dof, vertex in zip(dofs, element.vertices):
                    point = mesh.ngmesh.Points()[vertex.nr + 1]
                    x_value = float(point[0])
                    condition = (
                        boundary_values["left"]
                        if x_value <= midpoint
                        else boundary_values["right"]
                    )
                    vector[dof] = 0.0 if condition is None else _numeric_condition_at(condition, x_value)

        times = _time_values(interval, self.time_step)
        snapshots = [{"u": old}]
        iteration_counts, residuals = [], []
        converged = True
        for previous_time, current_time in zip(times[:-1], times[1:]):
            dt = float(current_time - previous_time)
            current = ngs.GridFunction(fes)
            current.vec.data = old.vec
            set_endpoint_values(current)
            step_residuals = []
            step_converged = False
            for iteration in range(1, self.max_iterations + 1):
                previous = _copy_grid_function(ngs, current, fes)
                trial = fes.TrialFunction()
                test = fes.TestFunction()
                bilinear = ngs.BilinearForm(fes)
                bilinear += (
                    trial * test / dt
                    + pde.nu * ngs.grad(trial) * ngs.grad(test)
                    + previous * ngs.grad(trial)[0] * test
                ) * ngs.dx
                linear = ngs.LinearForm(fes)
                linear += old * test / dt * ngs.dx
                bilinear.Assemble()
                linear.Assemble()
                _linear_solve(ngs, bilinear, linear, current, fes)
                set_endpoint_values(current)
                change = _relative_change(current, previous)
                step_residuals.append(change)
                if change <= self.tolerance:
                    step_converged = True
                    break
            snapshots.append({"u": current})
            old = current
            iteration_counts.append(iteration)
            residuals.append(step_residuals[-1] if step_residuals else math.inf)
            converged = converged and step_converged

        metadata = {
            "backend": "NGSolve/Netgen",
            "pde": pde.__class__.__name__,
            "fields": ["u"],
            "mesh_size": float(mesh_size),
            "mesh": {"dimension": 1, "elements": int(mesh.ne), "vertices": int(mesh.nv)},
            "spatial_coordinate": "x",
            "temporal_coordinate": "y",
            "time_values": times.tolist(),
            "time_step": float(np.min(np.diff(times))),
            "iterations": iteration_counts,
            "solver_residuals": residuals,
            "converged": converged,
        }
        return ReferenceSolution(
            area=pde_area,
            mesh=mesh,
            snapshots=snapshots,
            times=times,
            metadata=metadata,
            field_evaluator=_make_one_dimensional_evaluator(mesh),
            time_from_y=True,
        )


def _numeric_condition_at(condition, x, y=None):
    if _is_number(condition):
        return float(condition)
    variable, function = condition
    return _numeric_function_value(function, x if variable == "x" else (0.0 if y is None else y))


__all__ = [
    "ReferenceConfigurationError",
    "ReferenceGeometryError",
    "ReferenceSolver",
    "UnsupportedReferencePDE",
]
