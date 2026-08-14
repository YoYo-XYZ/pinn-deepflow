import numpy as np
import pytest
import ultraplot as plt

from deepflow.visualization import Visualizer


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _field_data():
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    return {
        "x": x.ravel(),
        "y": y.ravel(),
        "u": (x + y).ravel(),
    }


def test_plot_contour_renders_a_valid_2d_field():
    figure, axis = Visualizer(_field_data()).plot_contour("u", return_ax=True)

    assert figure is not None
    assert axis is not None


def test_plot_loss_curve_preserves_iteration_indices():
    visualizer = Visualizer({"total_loss": np.arange(10, dtype=float) + 1})

    _, axis = visualizer.plot_loss_curve(
        log_scale=False,
        start=4,
        end=8,
        return_ax=True,
    )

    np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(4, 8))


@pytest.mark.parametrize(
    "data, message",
    [
        (
            {"x": np.zeros(5), "y": np.arange(5.0), "u": np.arange(5.0)},
            "varying x and y",
        ),
        (
            {"x": np.arange(5.0), "y": np.zeros(5), "u": np.arange(5.0)},
            "varying x and y",
        ),
        (
            {"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0]), "u": np.array([0.0, 1.0])},
            "at least three",
        ),
    ],
)
def test_interpolate_rejects_invalid_2d_coordinates(data, message):
    with pytest.raises(ValueError, match=message):
        Visualizer(data)._interpolate("u")


def test_interpolate_rejects_mismatched_field_length():
    data = {
        "x": np.arange(4.0),
        "y": np.arange(4.0),
        "u": np.arange(3.0),
    }

    with pytest.raises(ValueError, match="coordinate length"):
        Visualizer(data)._interpolate("u")


def test_interpolate_rejects_collinear_samples():
    x = np.linspace(0, 1, 20)
    data = {"x": x, "y": x, "u": x}

    with pytest.raises(ValueError, match="non-collinear"):
        Visualizer(data)._interpolate("u")


def test_plot_color_does_not_accept_orientation():
    with pytest.raises(TypeError, match="orientation"):
        Visualizer(_field_data()).plot_color("u", orientation="horizontal")
