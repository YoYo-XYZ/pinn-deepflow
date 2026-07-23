"""Small helpers shared by the channel-flow benchmark scripts."""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
RESULTS_DIR = SCRIPT_DIR / "results"

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


def evaluation_grid(x_length, y_length, resolution):
    """Return a flattened, uniformly spaced ``(x, y)`` evaluation grid."""
    import numpy as np

    x = np.linspace(0, x_length, resolution[0])
    y = np.linspace(0, y_length, resolution[1])
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    return x_grid, y_grid, points
