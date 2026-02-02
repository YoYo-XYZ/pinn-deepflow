
import numpy as np
import matplotlib.pyplot as plt

def test_streamplot(nx, ny, min_x, max_x, min_y, max_y):
    print(f"Testing nx={nx}, ny={ny}, x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
    xi = np.linspace(min_x, max_x, nx)
    yi = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(xi, yi)
    U = np.ones_like(X)
    V = np.ones_like(Y)
    
    fig, ax = plt.subplots()
    try:
        ax.streamplot(X[0, :], Y[:, 0], U, V)
        print("Success")
    except ValueError as e:
        print(f"Failed: {e}")
        # manual check
        dx = np.diff(X[0, :])
        width = max_x - min_x
        expected_dx = width / (nx - 1)
        print(f"dx unique: {np.unique(dx)}")
        print(f"Expected dx: {expected_dx}")
        print(f"All close: {np.allclose(dx, expected_dx)}")
    plt.close(fig)

test_streamplot(100, 100, 0, 1, 0, 1)
test_streamplot(10, 10, 0.0003, 0.0004, 0, 1)
test_streamplot(50, 50, -0.5, 0.5, 0, 2.0)
