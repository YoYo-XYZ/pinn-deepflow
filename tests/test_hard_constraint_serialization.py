import torch

import deepflow as df


def test_hard_constraints_pickle_round_trip(tmp_path):
    bound = df.line_horizontal(0.0, [0.0, 1.0])
    bound.define_bc({"u": df.hard_constraint(3.0)})
    bound.sampling_line(4)
    bound.process_coordinates()

    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=4, length=1)
    model.apply_hard_constraints([bound])

    path = tmp_path / "hard_model.pkl"
    model.save_as_pickle(str(path))
    loaded = df.load_from_pickle(str(path))

    inputs = {
        "x": torch.tensor([0.0, 0.25, 1.0]),
        "y": torch.tensor([0.0, 0.5, 0.0]),
    }
    expected = model(inputs)["u"]
    actual = loaded(inputs)["u"]

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual[[0, 2]], torch.full((2,), 3.0))
