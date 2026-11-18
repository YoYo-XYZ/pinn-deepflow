from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytest

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def visualizer():
    x, y = np.meshgrid(np.linspace(0, 2, 5), np.linspace(0, 1, 4))
    x, y = x.ravel(), y.ravel()
    return df.Visualizer(
        {
            "x": x,
            "y": y,
            "u": x + 2 * y,
            "v": np.ones_like(x),
            "total_loss": np.array([8.0, 4.0, 2.0, 1.0]),
            "bc_loss": np.array([3.0, 2.0, 1.0, 0.5]),
        }
    )


def test_color_plot_uses_selected_coordinates_and_values_and_saves_png(visualizer):
    fig, ax = visualizer.plot_color("u", x_axis="y", y_axis="x", return_ax=True)
    scatter = ax.collections[0]
    np.testing.assert_array_equal(
        scatter.get_offsets(),
        np.column_stack((visualizer.data_dict["y"], visualizer.data_dict["x"])),
    )
    np.testing.assert_array_equal(scatter.get_array(), visualizer.data_dict["u"])
    assert ax.get_xlabel() == "y" and ax.get_ylabel() == "x"
    output = BytesIO()
    fig.savefig(output, format="png", dpi=30)
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_line_plot_returns_selected_field_data(visualizer):
    _, ax = visualizer.plot(x_axis="y", y_axis="u", return_ax=True)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), visualizer.data_dict["y"])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), visualizer.data_dict["u"])


def test_loss_curve_slices_actual_iterations_and_skips_missing_keys(visualizer):
    fig, ax = visualizer.plot_loss_curve(
        start=1, end=3, keys=("total_loss", "absent", "bc_loss"), return_ax=True
    )
    assert ax.get_yscale() == "log"
    assert len(ax.lines) == 2
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [4, 2])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [2, 1])
    fig.canvas.draw()


@pytest.mark.parametrize("kind", ["distribution", "contour", "streamline", "3d"])
def test_common_plot_types_render_on_headless_backend(kind, visualizer):
    if kind == "distribution":
        fig = visualizer.plot_distribution("u", bins=4)
    elif kind == "contour":
        fig = visualizer.plot_contour("u", levels=5)
    elif kind == "streamline":
        fig = visualizer.plot_streamline("u", "v", levels=5)
    else:
        fig = visualizer.plot("u")
    fig.canvas.draw()
    assert fig.axes


@pytest.mark.parametrize(
    "coordinates,message",
    [
        ({"x": [], "y": []}, "non-empty"),
        ({"x": [0, 1, 2], "y": [0, 1]}, "equal lengths"),
        ({"x": [0, 1], "y": [0, 1]}, "at least three"),
        ({"x": [0, 1, 2], "y": [0, np.nan, 1]}, "finite"),
        ({"x": [0, 0, 0], "y": [0, 1, 2]}, "varying"),
        ({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]}, "non-collinear"),
    ],
)
def test_contour_rejects_invalid_interpolation_data(coordinates, message):
    visualizer = df.Visualizer(coordinates | {"u": np.ones(len(coordinates["x"]))})
    with pytest.raises(ValueError, match=message):
        visualizer.plot_contour("u")


def test_contour_rejects_field_coordinate_length_mismatch(visualizer):
    visualizer.data_dict["u"] = np.ones(3)
    with pytest.raises(ValueError, match="must match the coordinate length"):
        visualizer.plot_contour("u")


def test_transient_evaluator_creates_a_small_animation(
    tmp_path, monkeypatch, polynomial_model
):
    import ultraplot

    monkeypatch.setattr(ultraplot, "show", lambda: None)
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_time([0, 1], sampling_scheme="uniform", expo_scaling=False)
    area.sampling_area([3, 3])
    evaluation = area.evaluate(polynomial_model)
    animation = evaluation.plot_animate(
        "u", range_t=[0.1, 0.3], dt=0.1, color_range=[0, 3]
    )
    path = tmp_path / "heat.gif"
    animation.save(str(path), writer="pillow", fps=2, dpi=30)
    assert path.read_bytes().startswith((b"GIF87a", b"GIF89a"))
