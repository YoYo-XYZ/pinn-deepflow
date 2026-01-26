from typing import List, Union, Dict, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

# Explicit imports are better for open source
from .neuralnetwork import PINN 
from .geometry import Area, Bound
from .visualization import Visualizer

class Evaluator(Visualizer):
    """
    Evaluates a PINN model against a given geometry and prepares data for visualization.
    """

    def __init__(self, pinns_model: PINN, geometry: Area|Bound):
        """
        Args:
            pinns_model: The physics-informed neural network model.
            geometry: The geometric domain (Area or Bound) to evaluate on.
        """
        self.model = pinns_model 
        self.geometry = geometry
        
        # Initialize internal state
        self.data_dict: Dict[str, Any] = {}
        self.is_postprocessed = False
        
        # If Visualizer requires data_dict immediately, initialize it with empty data
        # or handle the parent init carefully.
        # super().__init__({}) 

    def sampling_line(self, n_points: int) -> None:
        """Samples points along a line within the geometry."""
        self.geometry.sampling_line(n_points)
        self.postprocess()

    def sampling_area(self, res_list: List[int]) -> None:
        """Samples points within the area of the geometry."""
        self.geometry.sampling_area(res_list)
        self.postprocess()

    def define_time(self, time: Union[float, int]) -> None:
        """Defines time coordinates for transient problems."""
        self.geometry.define_time(time, "uniform", False)
        self.postprocess()

    def postprocess(self) -> None:
        """
        Aggregates model predictions, residuals, and coordinates, 
        then converts them to NumPy for visualization.
        """
        self.geometry.scheme = "uniform"
        self.geometry.expo_scaling = False
        self.geometry.process_coordinates()

        self._create_data_dict()
        self.is_postprocessed = True
        
        # Initialize the parent Visualizer with the processed data
        super().__init__(self.data_dict)

    def _create_data_dict(self) -> Dict[str, Any]:
        """Internal method to build the dictionary of fields."""
        # Ensure model is in eval mode to disable dropout/batchnorm updates
        self.model.eval()

        # 1. Base model outputs
        data_dict = self.geometry.process_model(self.model)

        # 2. Physics Residuals
        if self.geometry.physics_type == 'PDE':
            # Use .update() for dictionary merging (compatible with older python)
            data_dict[f"{self.geometry.physics_type}".lower() + "_residual"] = self.geometry.calc_loss_field(self.model)
            data_dict.update(self.geometry.PDE.var)
        
        elif self.geometry.physics_type in ['BC', 'IC']:
            data_dict[f"{self.geometry.physics_type}".lower() + "_residual"] = self.geometry.calc_loss_field(self.model)
        # 4. Coordinates
        data_dict['x'] = self.geometry.X
        data_dict['y'] = self.geometry.Y
        if self.geometry.range_t:
            data_dict['t'] = self.geometry.T

        # 5. Training History
        data_dict.update(self.model.loss_history)

        # 6. Normalize to NumPy
        self.data_dict = self._convert_to_numpy(data_dict)

        return self.data_dict

    def _convert_to_numpy(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Helper to safely convert dictionary values to NumPy arrays."""
        clean_dict = {}
        for key, value in data.items():
            try:
                if isinstance(value, list):
                    clean_dict[key] = np.array(value)
                elif isinstance(value, torch.Tensor):
                    clean_dict[key] = value.detach().cpu().numpy()
                elif isinstance(value, np.ndarray):
                    clean_dict[key] = value
                else:
                    # Fallback for scalars or other types
                    clean_dict[key] = np.array(value)
            except Exception as e:
                print(f"Warning: Could not convert key '{key}' to numpy. Error: {e}")
                clean_dict[key] = value
        return clean_dict
    
    def __getitem__(self, key: str) -> Any:
        return self.data_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.data_dict[key] = value

    def __str__(self):
        return f"Available data keys: {tuple(self.data_dict.keys())}"

    def plot_animate(self, color_axis: Union[str, Dict], x_axis: str = 'x', y_axis: str = 'y', range_t=None, dt=None, frame_interval = 10):
        """
        Creates an animation over time for the specified key(s).
        """
        import matplotlib.animation as animation

        key, cmap = self._normalize_args(color_axis)[0]

        # Prepare data to animate
        x = self.data_dict[x_axis]
        y = self.data_dict[y_axis]
        color_list = []
        time_list = list(np.arange(range_t[0], range_t[1], dt))
        for t in time_list:
            self.define_time(t)
            color_list.append(self.data_dict[key])
        max_val = np.max([np.max(c) for c in color_list])
        min_val = np.min([np.min(c) for c in color_list])

        fig, ax = plt.subplots()

        # Initialize figure
        scatter = ax.scatter(x, y, c=color_list[0], cmap=cmap, vmin=min_val, vmax=max_val, s=0.1)
        title = ax.set_title(f'{key} - Time: {time_list[0]:.3f}')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(key)
        
        def animate(frame):
            scatter.set_array(color_list[frame])
            title.set_text(f'{key} - Time: {time_list[frame]:.3f}')
            return scatter, title

        ani = animation.FuncAnimation(fig, animate, frames=len(time_list), interval=frame_interval, blit=True)
        plt.show()
        return ani
    