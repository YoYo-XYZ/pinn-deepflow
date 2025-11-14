from .Physics import PDE
from .Utility import calc_grad
import torch

class PhysicsAttach():
    def __init__(self):
        self.is_sampled = False
        self.physics_type = None
        self.range_t = None
        self.is_converged = False
#----------------------------------------------------------------------------------------------- usual conditions
    def define_bc(self, condition_dict: dict, range_t=None):
        """Define boundary conditions (BC)."""
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.range_t = range_t
        self.physics_type = "BC"
    
    def define_ic(self, condition_dict:dict, t=0.0):
        """Define initial conditions (IC)."""
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.t = t
        self.physics_type = "IC"
    
    def define_pde(self, PDE_class:PDE, range_t=None):
        """Define the PDE to enforce."""
        self.range_t = range_t
        self.PDE = PDE_class
        self.physics_type = "PDE"
#----------------------------------------------------------------------------------------------- input to PINNs
    def sampling_time(self, n_points, random=False):
        """Define the PDE to enforce."""
        if self.range_t is None:
            self.t = None
        else:
            if random:
                self.t = torch.empty(n_points).uniform_(self.range_t[0], self.range_t[1])
            else:
                self.t = torch.linspace(self.range_t[0], self.range_t[1])[:,None]

    def process_coordinates(self, device = 'cpu'):
        """Prepare coordinates data to be feed to PINNs"""
        self.X_ = self.X[:,None].to(device).requires_grad_()
        self.Y_ = self.Y[:,None].to(device).requires_grad_()

        if self.range_t is not None:
            self.T_ = self.t[:,None].to(device).requires_grad_()
            self.inputs_tensor_dict = {'x':self.X_,'y':self.Y_,'t':self.T_}
        else:
            self.inputs_tensor_dict = {'x':self.X_,'y':self.Y_, 't':None}

        if self.physics_type == "IC" or self.physics_type == "BC":
            target_output_tensor_dict = {}

            for key in self.condition_dict: #loop over condition
                if isinstance(self.condition_dict[key],(float,int)): #if condition is constant
                    target_output_tensor_dict[key] = self.condition_dict[key] * torch.ones_like(self.X_, device=device)
                else: #if condition varies function
                    variable_key = self.condition_dict[key][0]
                    func = self.condition_dict[key][1]
                    target_output_tensor_dict[key] = func(self.inputs_tensor_dict[variable_key].detach().clone())
            self.target_output_tensor_dict = target_output_tensor_dict

        return self.inputs_tensor_dict
#----------------------------------------------------------------------------------------------- process model's output and calculating loss
    def calc_output(self, model):
        """"Post-process the model's output to get predict (based from target_output_dict.)"""
        prediction_dict = model(self.inputs_tensor_dict)
        pred_dict = {}

        for key in self.target_output_tensor_dict:
            if '_' in key:
                key_split = key.split('_')
                pred_dict[key] = calc_grad(prediction_dict[key_split[0]], self.target_output_tensor_dict[key_split[1]])
            else:
                pred_dict[key] = prediction_dict[key]
        return pred_dict
    
    def calc_loss(self, model, loss_fn = torch.nn.MSELoss()):
        """Calculate loss from PINNs output"""
        if self.physics_type == "IC" or self.physics_type == "BC": 
            pred_dict = self.calc_output(model)

            loss = 0
            for key in pred_dict:
                loss += loss_fn(pred_dict[key], self.target_output_tensor_dict[key])
            return loss
        
        else:
            self.process_model(model)
            self.process_pde()
            return self.PDE.calc_loss()

    def calc_loss_field(self, model):
        """Calculate loss field from PINNs output"""
        if self.physics_type == "IC" or self.physics_type == "BC": 
            pred_dict = self.calc_output(model)

            loss_field = 0
            for key in pred_dict:
                loss_field += abs(pred_dict[key] - self.target_output_tensor_dict[key])
        
        elif self.physics_type == "PDE":
            self.process_model(model)
            self.process_pde()
            loss_field = self.PDE.calc_residual_field()

        self.loss_field = loss_field
        return loss_field

    def set_thereshold(self, loss = None, top_k_loss = None):
        self.loss_thereshold = loss
        self.top_k_loss_thereshold = top_k_loss

    def sampling_residual_based(self, top_k:int, model): #need more optimized
        """Add sampling points based on residual loss"""
        self.calc_loss_field(model) #TODO: NEED TO AVOID REPETETIVE CAL LOSS
        _, top_k_index = torch.topk(self.loss_field ,top_k, dim=0)
        top_k_index = top_k_index.flatten().to('cpu')
        self.X = torch.cat([self.X, self.X[top_k_index]])
        self.Y = torch.cat([self.Y, self.Y[top_k_index]])

        return self.X[top_k_index], self.Y[top_k_index]

#----------------------------------------------------------------------------------------------- process PDE related value
    def process_model(self, model):
        """feeds the inputs data to the model, returning model's output"""
        self.model_inputs = self.inputs_tensor_dict
        self.model_outputs = model(self.inputs_tensor_dict)
        return self.model_outputs
        
    def process_pde(self):
        self.PDE.calc(inputs_dict = self.model_inputs | self.model_outputs)

#-----------------------------------------------------------------------------------------------
    # def evaluate(self, model):
    #     return Evaluate(model, self)