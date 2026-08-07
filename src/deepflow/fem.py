"""FEM evaluation orchestration for :class:`ProblemDomain`."""

from typing import TYPE_CHECKING

import numpy as np
import torch

from .geometry import Area

if TYPE_CHECKING:
    from .domain import ProblemDomain
    from .evaluation import GroupEvaluator


def _coordinate_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _coordinates_match(public, processed) -> bool:
    if public is None or processed is None:
        return False
    public_array = _coordinate_array(public)
    processed_array = _coordinate_array(processed)
    return (
        public_array.shape == processed_array.shape
        and np.array_equal(public_array, processed_array)
    )


def _coordinate_shape(value):
    if value is None:
        return None
    return tuple(_coordinate_array(value).shape)


def _restore_time_coordinates(geometry, existing_time) -> None:
    geometry.t = existing_time
    geometry.T = existing_time
    if isinstance(existing_time, torch.Tensor):
        restored = existing_time.detach().clone()
        if isinstance(getattr(geometry, "X_", None), torch.Tensor):
            restored = restored.to(geometry.X_.device)
        geometry.T_ = restored.requires_grad_()
        if isinstance(getattr(geometry, "inputs_tensor_dict", None), dict):
            geometry.inputs_tensor_dict["t"] = geometry.T_
    else:
        geometry.T_ = existing_time


def _process_coordinates(geometry, preserve_time: bool) -> None:
    existing_time = None
    if preserve_time:
        for name in ("T", "t", "T_"):
            existing_time = getattr(geometry, name, None)
            if existing_time is not None:
                break
    existing_time_shape = _coordinate_shape(existing_time)

    geometry.process_coordinates()

    if (
        not preserve_time
        or existing_time is None
        or existing_time_shape is None
        or existing_time_shape != _coordinate_shape(geometry.X)
    ):
        return

    # ``PhysicsAttach.process_coordinates`` regenerates ``T`` when a range is
    # configured. Restore an already aligned time tensor when solve_fem is
    # reusing the caller's spatial coordinates.
    _restore_time_coordinates(geometry, existing_time)


def solve_fem(
    domain: "ProblemDomain",
    mesh_size=None,
    boundary_resolution=128,
    time_step=None,
    tolerance=1e-8,
    max_iterations=200,
    area_sampling_res=None,
    bound_sampling_res=None,
    *,
    reference_solver=None,
) -> "GroupEvaluator":
    """Solve a domain with the optional NGSolve FEM backend.

    Existing geometry coordinates are reused when sampling resolutions are
    omitted; supplied resolutions use the existing uniform samplers.
    """
    from .evaluation import GroupEvaluator

    bound_entries = GroupEvaluator._unique_entries(
        getattr(domain, "bound_list", [])
    )
    area_entries = GroupEvaluator._unique_entries(
        getattr(domain, "area_list", [])
    )
    sampled_geometry_ids = set()

    if bound_sampling_res is not None:
        resolutions = GroupEvaluator._normalize_resolutions(
            bound_sampling_res,
            len(bound_entries),
            "bound_sampling_res",
        )
        for (_, geometry), resolution in zip(bound_entries, resolutions):
            geometry.sampling_line(resolution, scheme="uniform")
            geometry.process_coordinates()
            sampled_geometry_ids.add(id(geometry))

    sampleable_area_entries = [
        (index, geometry)
        for index, geometry in area_entries
        if isinstance(geometry, Area)
    ]
    if area_sampling_res is not None:
        resolutions = GroupEvaluator._normalize_resolutions(
            area_sampling_res,
            len(sampleable_area_entries),
            "area_sampling_res",
            allow_area_pair=True,
        )
        for (_, geometry), resolution in zip(
            sampleable_area_entries, resolutions
        ):
            geometry.sampling_area(resolution, scheme="uniform")
            geometry.process_coordinates()
            sampled_geometry_ids.add(id(geometry))

    geometries = [
        geometry
        for _, geometry in bound_entries + area_entries
    ]

    missing = []
    for category, entries in (
        ("bound", bound_entries),
        ("area", area_entries),
    ):
        for index, geometry in entries:
            if (
                getattr(geometry, "X", None) is None
                or getattr(geometry, "Y", None) is None
            ):
                missing.append(
                    f"{category}[{index}] ({type(geometry).__name__})"
                )

    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Cannot evaluate unsampled domain geometries: "
            f"{joined}. Sample every geometry before calling solve_fem()."
        )

    for geometry in geometries:
        needs_processing = (
            id(geometry) in sampled_geometry_ids
            or not _coordinates_match(
                getattr(geometry, "X", None),
                getattr(geometry, "X_", None),
            )
            or not _coordinates_match(
                getattr(geometry, "Y", None),
                getattr(geometry, "Y_", None),
            )
        )
        has_time = any(
            getattr(geometry, name, None) is not None
            for name in ("T", "t", "T_")
        )
        if (
            not has_time
            and getattr(geometry, "range_t", None) is not None
        ):
            needs_processing = True
        if needs_processing:
            _process_coordinates(
                geometry,
                preserve_time=id(geometry) not in sampled_geometry_ids,
            )

    solver_class = reference_solver
    if solver_class is None:
        from .reference import ReferenceSolver as solver_class

    reference_solution = solver_class(
        mesh_size=mesh_size,
        boundary_resolution=boundary_resolution,
        time_step=time_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).solve(domain)
    return GroupEvaluator(
        None,
        domain,
        reference_solution=reference_solution,
    )
