import torch
import numpy as np
from .Network import *
from .Geometry import Area, Bound
from .Visualize import Visualizer
 
class Evaluate(Visualizer):
    def __init__(self, pinns_model:PINN, geometry:Area|Bound):
        self.model = pinns_model.cpu()
        self.data_dict = {}
        self.geometry = geometry
        self.is_preprocessed = False

    def sampling_line(self, n_points:int):
        self.geometry.sampling_line(n_points)
        self.geometry.process_coordinates()
        self.postprocess()

    def sampling_area(self, x_res:int, y_res:int):
        self.geometry.sampling_area([x_res, y_res])
        self.geometry.process_coordinates()
        self.postprocess()

    def postprocess(self):
        self._create_data_dict()
        print(f"available_data: {tuple(self.data_dict.keys())}")
        self.is_postprocessed = True
        Visualizer.__init__(self, self.data_dict)

    def _create_data_dict(self):
        # create data_dict
        data_dict = {}
        data_dict = data_dict | self.geometry.process_model(self.model)

        # velocity magnitude
        if "u" in data_dict and "v" in data_dict:
            data_dict["velocity_magnitude"] = torch.sqrt(data_dict["u"]**2 + data_dict["v"]**2)
        # PDE residual
        if self.geometry.physics_type == 'PDE':
            data_dict = data_dict |self.geometry.PDE.var
            data_dict[f"{self.geometry.physics_type} residual"] = self.geometry.calc_loss_field(self.model)
        # IC/BC residual
        if self.geometry.physics_type == 'BC' or self.geometry.physics_type == 'IC':
            data_dict[f"{self.geometry.physics_type} residual"] = self.geometry.calc_loss_field(self.model)

        # X, Y coordinates
        data_dict['x'] = self.geometry.X
        data_dict['y'] = self.geometry.Y

        # model
        data_dict = data_dict | self.model.loss_history_dict

        # normalize to numpy
        for key in data_dict:
            if isinstance(data_dict[key], list):
                data_dict[key] = np.array(data_dict[key])
            else:
                data_dict[key] = data_dict[key].detach().flatten().cpu().numpy()

        # store in self.data_dict
        self.data_dict = data_dict
        return data_dict
