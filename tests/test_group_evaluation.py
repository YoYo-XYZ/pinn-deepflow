import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import deepflow as df
import deepflow.utility as _df_util
from deepflow.visualization import Visualizer


@pytest.fixture(autouse=True)
def _cpu_device():
    original_device = _df_util.device
    original_dtype = df.get_dtype()
    _df_util.device = "cpu"
    try:
        yield
    finally:
        _df_util.device = original_device
        df.set_dtype(original_dtype)


def _model(input_vars=None):
    return df.PINN(
        input_vars=input_vars or ["x", "y"],
        output_vars=["u"],
        width=4,
        length=1,
    )


def _sample_unique_bounds(domain, n_points=5):
    seen = set()
    for bound in domain.bound_list:
        if id(bound) in seen:
            continue
        seen.add(id(bound))
        bound.sampling_line(n_points)
        bound.process_coordinates()


def _sample_unique_areas(domain, resolution=(4, 4)):
    seen = set()
    for area in domain.area_list:
        if id(area) in seen:
            continue
        seen.add(id(area))
        if isinstance(area, df.Area):
            area.sampling_area(list(resolution))
            area.process_coordinates()
        else:
            area.process_coordinates()


def test_problem_domain_evaluate_returns_structured_group():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    area.define_pde(df.pde.BurgersEquation1D(nu=0.01))
    for bound in domain.bound_list:
        bound.define_bc({"u": 0})

    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)

    group = domain.evaluate(_model())

    assert isinstance(group, df.GroupEvaluator)
    assert len(group.area_evaluators) == 1
    assert len(group.bound_evaluators) == 4
    assert group.area_evaluators[0].geometry is area
    assert "pde_residual" in group.area_evaluators[0].data_dict
    assert "bc_residual" in group.bound_evaluators[0].data_dict
    assert list(group) == group.bound_evaluators + group.area_evaluators


def test_group_evaluator_deduplicates_repeated_geometry_references():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area, area.bound_list)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)

    group = domain.evaluate(_model())

    assert len(group.area_evaluators) == 1
    assert len(group.bound_evaluators) == 4
    assert group.get_evaluator(area) is group.area_evaluators[0]
    assert group.get_evaluator(area.bound_list[0]) is group.bound_evaluators[0]


def test_group_evaluator_rejects_unsampled_geometries():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)

    with pytest.raises(ValueError, match="area\\[0\\].*Area"):
        domain.evaluate(_model())


def test_group_evaluator_handles_custom_data_without_sampling():
    custom = df.custom_data(
        {
            "x": torch.linspace(0.0, 1.0, 4),
            "y": torch.linspace(0.0, 1.0, 4),
        }
    )
    group = df.domain(custom).evaluate(_model())

    assert len(group.area_evaluators) == 1
    assert group.area_evaluators[0].geometry is custom
    assert group.area_evaluators[0].data_dict["x"].shape == (4,)


def test_group_sampling_helpers_refresh_child_evaluators():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)
    group = domain.evaluate(_model())

    group.sampling_line([7, 8, 9, 10])
    group.sampling_area([6, 5])

    assert [
        evaluator.geometry.X.shape[0]
        for evaluator in group.bound_evaluators
    ] == [7, 8, 9, 10]
    assert group.area_evaluators[0].geometry.X.shape[0] == 30
    assert [
        evaluator.data_dict["x"].shape[0]
        for evaluator in group.bound_evaluators
    ] == [7, 8, 9, 10]
    assert group.area_evaluators[0].data_dict["x"].shape[0] == 30


def test_group_define_time_broadcasts_to_all_children():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)
    group = domain.evaluate(_model())

    group.define_time([0.0, 1.0], sampling_scheme="random", expo_scaling=True)

    for evaluator in group:
        geometry = evaluator.geometry
        assert geometry.scheme == "random"
        assert geometry.expo_scaling is True
        assert geometry.T_.shape == geometry.X_.shape
        assert "t" in evaluator.data_dict


def test_group_refresh_recomputes_child_data_after_resampling():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain, resolution=(3, 3))
    group = domain.evaluate(_model())

    area.sampling_area([5, 5])
    area.process_coordinates()
    group.refresh()

    assert group.area_evaluators[0].data_dict["x"].shape[0] == 25


def test_group_plot_delegation_uses_selected_geometry(monkeypatch):
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)
    group = domain.evaluate(_model())

    calls = []

    def fake_plot_color(self, *args, **kwargs):
        calls.append((self.geometry, args, kwargs))
        return "plot-result"

    monkeypatch.setattr(df.Evaluator, "plot_color", fake_plot_color)

    result = group.plot_color("u", geometry=area, s=1)

    assert result == "plot-result"
    assert calls == [(area, ("u",), {"s": 1})]


def test_group_plot_color_aggregates_all_compatible_children(monkeypatch):
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)
    group = domain.evaluate(_model())

    captured = {}

    def fake_plot_color(self, **kwargs):
        captured["data"] = self.data_dict
        captured["kwargs"] = kwargs
        return "aggregate-plot"

    monkeypatch.setattr(Visualizer, "plot_color", fake_plot_color)

    result = group.plot_color("u", s=3, return_ax=True)

    expected_points = sum(
        evaluator.data_dict["x"].shape[0]
        for evaluator in group
    )
    assert result == "aggregate-plot"
    assert captured["data"]["x"].shape == (expected_points,)
    assert captured["data"]["y"].shape == (expected_points,)
    assert captured["data"]["u"].shape == (expected_points,)
    assert captured["kwargs"] == {
        "color_axis": "u",
        "x_axis": "x",
        "y_axis": "y",
        "cmap": "viridis",
        "s": 3,
        "orientation": "vertical",
        "return_ax": True,
    }
    assert not hasattr(group, "data_dict")


def test_group_plot_color_skips_children_missing_color_field(monkeypatch):
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)
    group = domain.evaluate(_model())

    missing_evaluator = group.bound_evaluators[0]
    missing_points = missing_evaluator.data_dict["x"].shape[0]
    del missing_evaluator.data_dict["u"]

    captured = {}

    def fake_plot_color(self, **kwargs):
        captured["data"] = self.data_dict
        return "aggregate-plot"

    monkeypatch.setattr(Visualizer, "plot_color", fake_plot_color)

    assert group.plot_color("u") == "aggregate-plot"
    expected_points = sum(
        evaluator.data_dict["x"].shape[0]
        for evaluator in group
    ) - missing_points
    assert captured["data"]["u"].shape == (expected_points,)


def test_group_plot_color_rejects_missing_field_everywhere():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)
    group = domain.evaluate(_model())

    for evaluator in group:
        del evaluator.data_dict["u"]

    with pytest.raises(KeyError, match="No evaluated geometry"):
        group.plot_color("u")


def test_group_plot_color_rejects_mismatched_child_lengths():
    area = df.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    _sample_unique_bounds(domain)
    _sample_unique_areas(domain)
    group = domain.evaluate(_model())

    evaluator = group.bound_evaluators[0]
    evaluator.data_dict["u"] = evaluator.data_dict["u"][:-1]

    with pytest.raises(ValueError, match="Cannot aggregate plot_color"):
        group.plot_color("u")
