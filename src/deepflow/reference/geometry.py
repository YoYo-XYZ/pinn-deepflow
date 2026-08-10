"""Conversion of DeepFlow boundaries into a Netgen 2-D geometry."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..geometry import Area, Bound


class ReferenceGeometryError(ValueError):
    """Raised when a DeepFlow geometry cannot be meshed by the reference backend."""


@dataclass
class _Edge:
    bound: Bound
    label: str
    points: np.ndarray


@dataclass
class _Loop:
    edges: List[_Edge]
    hole: bool


@dataclass
class MeshGeometry:
    """Netgen geometry plus the labels used by the FEM solver."""

    geometry: object
    mesh: object
    loops: List[_Loop]
    labels_by_bound: Dict[int, str]
    boundary_info: Dict[str, dict]
    points_by_label: Dict[str, np.ndarray]
    area: Area


def _as_points(x, y, bound: Bound) -> np.ndarray:
    try:
        x_array = np.asarray(x.detach().cpu(), dtype=float)
        y_array = np.asarray(y.detach().cpu(), dtype=float)
    except AttributeError:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)

    points = np.column_stack((x_array.reshape(-1), y_array.reshape(-1)))
    if points.shape[0] < 2 or not np.isfinite(points).all():
        raise ReferenceGeometryError(
            "Boundary sampling produced fewer than two finite points for "
            f"{bound!r}."
        )

    scale = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    tolerance = scale * 1.0e-12
    keep = [0]
    for index in range(1, len(points)):
        if np.linalg.norm(points[index] - points[keep[-1]]) > tolerance:
            keep.append(index)
    points = points[keep]
    if len(points) < 2:
        raise ReferenceGeometryError(
            "A boundary is degenerate after sampling; explicit non-degenerate "
            "boundary curves are required."
        )
    return points


def _sample_bound(bound: Bound, resolution: int) -> np.ndarray:
    if not isinstance(bound, Bound):
        raise ReferenceGeometryError(
            "Reference geometry boundaries must be deepflow.geometry.Bound objects."
        )
    try:
        x, y = bound.sampling_line(resolution, scheme="uniform")
    except Exception as exc:
        raise ReferenceGeometryError(
            "Could not sample an explicit boundary curve.  An arbitrary "
            "contains_fn without usable Bound objects is not supported."
        ) from exc
    return _as_points(x, y, bound)


def _close(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    return float(np.linalg.norm(a - b)) <= tolerance


def _stitch(edges: Sequence[_Edge], tolerance: float) -> List[List[_Edge]]:
    """Stitch unoriented sampled curves into closed loops."""
    remaining = list(range(len(edges)))
    loops: List[List[_Edge]] = []

    while remaining:
        first_index = remaining.pop(0)
        first = edges[first_index]
        current = _Edge(first.bound, first.label, first.points.copy())
        loop = [current]
        start = current.points[0]
        end = current.points[-1]

        while not _close(end, start, tolerance):
            candidates = []
            for position, index in enumerate(remaining):
                candidate = edges[index]
                if _close(candidate.points[0], end, tolerance):
                    candidates.append((position, False))
                if _close(candidate.points[-1], end, tolerance):
                    candidates.append((position, True))

            if not candidates:
                raise ReferenceGeometryError(
                    "Boundary curves do not form a closed loop; check for an "
                    "open or ambiguous Bound representation."
                )

            # The first match is deterministic.  Ambiguous branches are not
            # silently joined: they usually indicate a disconnected union.
            if len(candidates) > 1:
                unique_edges = {remaining[position] for position, _ in candidates}
                if len(unique_edges) > 1:
                    raise ReferenceGeometryError(
                        "Boundary curves have an ambiguous junction.  The "
                        "reference backend requires one connected outer loop "
                        "and explicitly represented holes."
                    )

            position, reverse = candidates[0]
            index = remaining.pop(position)
            candidate = edges[index]
            points = candidate.points[::-1].copy() if reverse else candidate.points.copy()
            loop.append(_Edge(candidate.bound, candidate.label, points))
            end = points[-1]

            if len(loop) > len(edges):
                raise ReferenceGeometryError("Could not stitch boundary loops safely.")

        loop[-1].points[-1] = start
        loops.append(loop)

    return loops


def _loop_points(loop: _Loop) -> np.ndarray:
    points = [edge.points[:-1] for edge in loop.edges]
    points.append(loop.edges[-1].points[-1][None, :])
    return np.concatenate(points, axis=0)


def _signed_area(points: np.ndarray) -> float:
    closed = np.vstack((points, points[0]))
    return 0.5 * float(
        np.sum(closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1])
    )


def _reverse_loop(loop: _Loop) -> _Loop:
    return _Loop(
        edges=[_Edge(edge.bound, edge.label, edge.points[::-1].copy()) for edge in loop.edges[::-1]],
        hole=loop.hole,
    )


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = point
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        x1, y1 = first
        x2, y2 = second
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersection:
                inside = not inside
    return inside


def _representative_point(points: np.ndarray) -> np.ndarray:
    candidate = np.mean(points, axis=0)
    if _point_in_polygon(candidate, points):
        return candidate

    lower = points.min(axis=0)
    upper = points.max(axis=0)
    for fraction_x in np.linspace(0.2, 0.8, 7):
        for fraction_y in np.linspace(0.2, 0.8, 7):
            candidate = lower + (upper - lower) * [fraction_x, fraction_y]
            if _point_in_polygon(candidate, points):
                return candidate
    return np.mean(points, axis=0)


class NetgenGeometryAdapter:
    """Build a Netgen mesh from one explicit DeepFlow ``Area``."""

    def __init__(self, boundary_resolution: int = 128):
        if not isinstance(boundary_resolution, int) or boundary_resolution < 4:
            raise ValueError("boundary_resolution must be an integer >= 4")
        self.boundary_resolution = boundary_resolution

    def _make_edges(self, bounds: Sequence[Bound], prefix: str, hole: bool) -> List[_Edge]:
        edges = []
        for index, bound in enumerate(bounds):
            label = f"{prefix}_{index:04d}"
            edges.append(
                _Edge(bound, label, _sample_bound(bound, self.boundary_resolution))
            )
        return edges

    def _loops_for_bounds(
        self, bounds: Sequence[Bound], prefix: str, hole: bool, tolerance: float
    ) -> List[_Loop]:
        if not bounds:
            return []
        edges = self._make_edges(bounds, prefix, hole)
        loops = [_Loop(loop, hole) for loop in _stitch(edges, tolerance)]
        for loop_index, loop in enumerate(loops):
            area = _signed_area(_loop_points(loop))
            if abs(area) <= tolerance * tolerance:
                raise ReferenceGeometryError("A boundary loop has zero area.")
            # Netgen uses the material on the left side of a positively
            # oriented outer loop.  Holes therefore use the opposite winding.
            if (not hole and area < 0) or (hole and area > 0):
                loop = _reverse_loop(loop)
            # Replace the loop in-place so callers retain deterministic order.
            loops[loop_index] = loop
        return loops

    def build(self, area: Area, ngsolve_module, mesh_size: float) -> MeshGeometry:
        if not isinstance(area, Area):
            raise TypeError("ReferenceSolver requires a DeepFlow Area geometry.")
        if not area.bound_list:
            raise ReferenceGeometryError(
                "The reference backend requires explicit Bound objects; an "
                "arbitrary contains_fn is not enough to generate a mesh."
            )
        if mesh_size <= 0:
            raise ValueError("mesh_size must be positive")

        ranges = area.ranges
        scale = max(
            abs(float(ranges[0][1] - ranges[0][0])),
            abs(float(ranges[1][1] - ranges[1][0])),
            1.0,
        )
        tolerance = max(scale * 1.0e-7, 1.0e-10)
        outer_loops = self._loops_for_bounds(
            area.bound_list, "outer", False, tolerance
        )
        if len(outer_loops) != 1:
            raise ReferenceGeometryError(
                "The PDE area must have exactly one connected outer boundary. "
                "Use negative_bound_list for holes."
            )
        hole_bounds = list(area.negative_bound_list or [])
        hole_loops = self._loops_for_bounds(hole_bounds, "hole", True, tolerance)
        loops = outer_loops + hole_loops

        outer_points = _loop_points(outer_loops[0])[:-1]
        for loop in hole_loops:
            representative = _representative_point(_loop_points(loop)[:-1])
            if not _point_in_polygon(representative, outer_points):
                raise ReferenceGeometryError("A hole boundary lies outside the outer domain.")

        from netgen.geom2d import SplineGeometry

        geometry = SplineGeometry()
        point_ids = {}

        def point_id(point):
            key = tuple(np.round(np.asarray(point, dtype=float) / tolerance).astype(np.int64))
            if key not in point_ids:
                point_ids[key] = geometry.AppendPoint(float(point[0]), float(point[1]))
            return point_ids[key]

        for loop in loops:
            for edge in loop.edges:
                points = edge.points
                for first, second in zip(points[:-1], points[1:]):
                    if np.linalg.norm(first - second) <= tolerance:
                        continue
                    geometry.Append(
                        ["line", point_id(first), point_id(second)], bc=edge.label
                    )

        try:
            mesh = ngsolve_module.Mesh(geometry.GenerateMesh(maxh=float(mesh_size)))
        except Exception as exc:
            raise ReferenceGeometryError(
                "Netgen could not generate a mesh from the explicit boundary loops."
            ) from exc

        labels_by_bound = {}
        boundary_info = {}
        points_by_label = {}
        for loop in loops:
            for edge in loop.edges:
                labels_by_bound[id(edge.bound)] = edge.label
                boundary_info[edge.label] = {
                    "hole": bool(loop.hole),
                    "bound_id": id(edge.bound),
                }
                points_by_label[edge.label] = edge.points.copy()

        return MeshGeometry(
            geometry=geometry,
            mesh=mesh,
            loops=loops,
            labels_by_bound=labels_by_bound,
            boundary_info=boundary_info,
            points_by_label=points_by_label,
            area=area,
        )
