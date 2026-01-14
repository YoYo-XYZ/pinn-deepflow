from typing import List, Union, Dict, Any, Optional

import torch
import numpy as np

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
        self.geometry.process_coordinates()
        self.postprocess()

    def sampling_area(self, res_list: List[int]) -> None:
        """Samples points within the area of the geometry."""
        self.geometry.sampling_area(res_list)
        self.geometry.process_coordinates()
        self.postprocess()

    def postprocess(self) -> None:
        """
        Aggregates model predictions, residuals, and coordinates, 
        then converts them to NumPy for visualization.
        """
        self._create_data_dict()
        print(f"Available data keys: {tuple(self.data_dict.keys())}")
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
            data_dict[f"{self.geometry.physics_type} residual"] = self.geometry.calc_loss_field(self.model)
            data_dict.update(self.geometry.PDE.var)
        
        elif self.geometry.physics_type in ('BC', 'IC'):
            data_dict[f"{self.geometry.physics_type} residual"] = self.geometry.calc_loss_field(self.model)

        # 4. Coordinates
        data_dict['x'] = self.geometry.X
        data_dict['y'] = self.geometry.Y

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