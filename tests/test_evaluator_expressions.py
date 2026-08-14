import os
import sys

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import deepflow as df  # noqa: E402
import deepflow.utility as _df_util  # noqa: E402


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


def _manual_evaluator(data=None, *, postprocessed=True):
    evaluator = df.Evaluator(None, None)
    evaluator.data_dict = {} if data is None else dict(data)
    evaluator.is_postprocessed = postprocessed
    return evaluator


def test_lazy_expression_computes_arithmetic_and_numpy_ufuncs():
    evaluator = _manual_evaluator(
        {
            "u": np.array([3.0, 4.0]),
            "v": np.array([4.0, 3.0]),
        }
    )
    fields = evaluator.expr

    fields["magnitude"] = (fields["u"] ** 2 + fields["v"] ** 2) ** 0.5
    fields["magnitude_ufunc"] = np.hypot(fields["u"], fields["v"])

    np.testing.assert_allclose(evaluator["magnitude"], [5.0, 5.0])
    np.testing.assert_allclose(evaluator["magnitude_ufunc"], [5.0, 5.0])
    assert "u" in repr(fields["u"] ** 2)


def test_lazy_expression_supports_reverse_operators_and_where():
    evaluator = _manual_evaluator({"u": np.array([-2.0, 3.0])})
    fields = evaluator.expr

    fields["shifted"] = 2 + fields["u"]
    fields["ratio"] = 10 / (fields["u"] + 3)
    fields["positive"] = np.where(fields["u"] > 0, fields["u"], 0)
    fields["absolute"] = abs(fields["u"])

    np.testing.assert_allclose(evaluator["shifted"], [0.0, 5.0])
    np.testing.assert_allclose(evaluator["ratio"], [10.0, 10.0 / 6.0])
    np.testing.assert_allclose(evaluator["positive"], [0.0, 3.0])
    np.testing.assert_allclose(evaluator["absolute"], [2.0, 3.0])


def test_lazy_expression_supports_chained_fields_and_refresh():
    evaluator = _manual_evaluator(
        {"u": np.array([3.0, 4.0]), "v": np.array([4.0, 3.0])}
    )
    fields = evaluator.expr
    fields["magnitude"] = np.hypot(fields["u"], fields["v"])
    fields["energy"] = 0.5 * fields["magnitude"] ** 2

    evaluator.data_dict = {
        "u": np.array([5.0]),
        "v": np.array([12.0]),
    }
    evaluator._refresh_derived_fields()

    np.testing.assert_allclose(evaluator["magnitude"], [13.0])
    np.testing.assert_allclose(evaluator["energy"], [84.5])


def test_lazy_expression_can_be_registered_before_data_exists():
    evaluator = _manual_evaluator(postprocessed=False)
    fields = evaluator.expr
    fields["magnitude"] = np.hypot(fields["u"], fields["v"])

    evaluator.data_dict = {
        "u": np.array([3.0]),
        "v": np.array([4.0]),
    }
    evaluator._refresh_derived_fields()

    np.testing.assert_allclose(evaluator["magnitude"], [5.0])


def test_lazy_expression_rejects_invalid_dependencies_and_cycles():
    evaluator = _manual_evaluator({"u": np.array([1.0])})

    with pytest.raises(KeyError, match="missing data key 'missing'"):
        evaluator.expr["bad"] = evaluator.expr["missing"] + 1

    evaluator = _manual_evaluator(postprocessed=False)
    evaluator.expr["first"] = evaluator.expr["second"] + 1
    evaluator.expr["second"] = evaluator.expr["first"] + 1
    evaluator.data_dict = {"u": np.array([1.0])}
    evaluator.is_postprocessed = True

    with pytest.raises(ValueError, match="Cyclic derived-field dependency"):
        evaluator._refresh_derived_fields()


def test_lazy_expression_rejects_collisions_cross_evaluator_and_unsupported_numpy():
    evaluator = _manual_evaluator({"u": np.array([1.0])})
    other = _manual_evaluator({"u": np.array([2.0])})

    with pytest.raises(ValueError, match="already a base data key"):
        evaluator.expr["u"] = evaluator.expr["u"] + 1

    with pytest.raises(ValueError, match="different Evaluator"):
        evaluator.expr["cross"] = other.expr["u"] + 1

    with pytest.raises(TypeError, match="not supported"):
        np.mean(evaluator.expr["u"])


def test_direct_assignment_removes_persistent_expression():
    evaluator = _manual_evaluator({"u": np.array([1.0, 2.0])})
    fields = evaluator.expr
    fields["double"] = 2 * fields["u"]

    evaluator["double"] = np.array([9.0, 9.0])
    evaluator._refresh_derived_fields()

    np.testing.assert_allclose(evaluator["double"], [9.0, 9.0])
    with pytest.raises(KeyError):
        del fields["double"]


def test_lazy_expression_works_with_real_evaluator_after_resampling():
    area = df.rectangle([0, 1], [0, 1])
    area.sampling_area([3, 3])
    area.process_coordinates()
    model = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u", "v"],
        width=4,
        length=1,
    )
    evaluator = area.evaluate(model)
    fields = evaluator.expr
    fields["v_magnitude"] = np.hypot(fields["u"], fields["v"])

    evaluator.sampling_area([4, 4])

    assert evaluator["v_magnitude"].shape == evaluator["u"].shape
    np.testing.assert_allclose(
        evaluator["v_magnitude"],
        np.hypot(evaluator["u"], evaluator["v"]),
    )


class _FakeReferenceSolution:
    fields = ("u", "v")
    is_transient = False
    time_from_y = False
    metadata = {"fields": ["u", "v"]}

    def evaluate(self, x, y, t=None, fields=None):
        values = {}
        for index, name in enumerate(fields or self.fields):
            values[name] = np.asarray(x, dtype=float) + index
        return values


def test_lazy_expression_works_with_reference_evaluator():
    custom = df.custom_data(
        {
            "x": torch.linspace(0.0, 1.0, 4),
            "y": torch.linspace(0.0, 1.0, 4),
        }
    )
    evaluator = df.ReferenceEvaluator(_FakeReferenceSolution(), custom)
    fields = evaluator.expr
    fields["magnitude"] = np.hypot(fields["u_ref"], fields["v_ref"])

    evaluator.postprocess()

    np.testing.assert_allclose(
        evaluator["magnitude"],
        np.hypot(evaluator["u_ref"], evaluator["v_ref"]),
    )


def test_deleting_lazy_expression_removes_materialized_field():
    evaluator = _manual_evaluator({"u": np.array([1.0])})
    fields = evaluator.expr
    fields["double"] = 2 * fields["u"]

    del fields["double"]

    assert "double" not in evaluator.data_dict
