import sys
import torch
import torch.nn as nn
import copy

class PINN(nn.Module):
    """
    Physics-Informed Neural Network model.

    A feedforward neural network that takes {'x':, "y", t)
    and outputs the fluid velocity components u, v and pressure p.
    """
    def __init__(self, width, length, inputs = ['x','y'], outputs = ['u','v','p']):
        super().__init__()

        self.input_key = inputs
        self.output_key = outputs
        self.input_param = len(inputs)
        self.output_param = len(outputs)
        self.is_steady = not 't' in inputs

        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_param, width))
        layers.append(nn.Tanh())
        # Hidden layers
        for _ in range(length):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        # Output layer
        layers.append(nn.Linear(width, self.output_param))
        self.net = nn.Sequential(*layers)

        if self.is_steady:
            self.loss_history_dict = {'total_loss':[], 'bc_loss':[], 'pde_loss':[]}
        else:
            self.loss_history_dict = {'total_loss':[], 'bc_loss':[], 'ic_loss':[], 'pde_loss':[]}

    def forward(self, inputs_dict):
        """Forward pass through the network."""
        if self.is_steady:
            input_tensor = torch.cat([inputs_dict[key] for key in self.input_key], dim=1)
        else:
            input_tensor = torch.cat([inputs_dict[key] for key in self.output_key], dim=1)

        pred = self.net(input_tensor)
        output_dict = {}
        for i, key in enumerate(self.output_key):
            output_dict[key] = pred[:, i:i+1]
        return output_dict
    
    def show_updates(self):
        print(f"epoch {len(self.loss_history_dict['total_loss'])}, " + ", ".join(f"{key}: {value[-1]:.5f}" for key, value in self.loss_history_dict.items()))

#-----------------------------------------------------------------------
class NetworkTrainer():
    def __init__(self):
        self.optimizer_choice = {"Adam":None, "LBFGS":None}
    
    @staticmethod
    def record_loss(loss_hist_dict, loss_dict):
        for key in loss_dict:
            loss_hist_dict[key].append(loss_dict[key].item())
        return loss_hist_dict
    
    @staticmethod
    def train_adam(model, learning_rate, epochs, calc_loss, print_every=50, threshold_loss=None, device= 'cpu'):
        model = copy.deepcopy(model.to(device))
        if device == 'cuda' and sys.platform.startswith('linux'):
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True, backend="inductor")
            print('model is compiled for cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss_dict = calc_loss(model)
            loss_dict['total_loss'].backward()
            optimizer.step()

            model.loss_history_dict = NetworkTrainer.record_loss(model.loss_history_dict, loss_dict)

            if epoch % print_every == 0:
                model.show_updates()

            if model.loss_history_dict['total_loss'][-1] < threshold_loss:
                print(f"Training stopped at epoch {epoch} as total loss reached the threshold of {threshold_loss}.")
                break
        return model.to(device)

    @staticmethod
    def train_lbfgs(model, epochs, calc_loss, print_every=50, threshold_loss=None, device= 'cpu'):
        model = copy.deepcopy(model.to(device))
        if device == 'cuda' and sys.platform.startswith('linux'):
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True, backend="inductor")
            print('model is compiled for cuda')

        optimizer = torch.optim.LBFGS(model.parameters(), history_size=20, max_iter=10, line_search_fn="strong_wolfe")
        for epoch in range(epochs):
            # Create a closure that captures loss_dict
            loss_dict_container = {}
            
            def closure():
                optimizer.zero_grad()
                loss_dict = calc_loss(model)
                loss_dict['total_loss'].backward()
                # Store loss_dict in the container
                loss_dict_container['loss_dict'] = loss_dict
                return loss_dict['total_loss']
            
            optimizer.step(closure)
            
            # Retrieve the loss_dict from the container
            loss_dict = loss_dict_container['loss_dict']
            model.loss_history_dict = NetworkTrainer.record_loss(model.loss_history_dict, loss_dict)
            
            if epoch % print_every == 0:
                model.show_updates()
            
            if threshold_loss is not None and model.loss_history_dict['total_loss'][-1] < threshold_loss:
                print(f"Training stopped at epoch {epoch} as total loss reached the threshold of {threshold_loss}.")
                break
        
        return model


