import deepflow as df
line_ic = df.geometry.line_horizontal(y=0, range_x=[0,0.1])
point1 = df.geometry.point(x=0.01, y=0)
point2 = df.geometry.point(x=0.09, y=0)
point3 = df.geometry.point(x=0.05, y=0)
domain = df.domain(line_ic, point1, point2, point3)
domain.show_setup()
import torch
class Beambending(df.PDE):
    def __init__(self, E=2.2*10**9, I=0.0000001, F=1000):
        super().__init__()
        self.E = E
        self.I = I
        self.F = F

    def M(self, x):
        mask = (x > 0.01)
        mask2 = (x < 0.09)
        return torch.where(mask, self.F/2*(x-0.01), torch.where(mask2, self.F/2*(0.09-x), x*0.0))

    def compute_residuals(self, inputs_dict):
        x = inputs_dict['x']
        y = inputs_dict['u']

        y_x = df.calc_grad(y, x)
        y_xx = df.calc_grad(y_x, x)
        self.residual_fields = ((y_xx - self.M(x))/(self.E*self.I),)
# Define PDE
domain.bound_list[0].define_pde(Beambending())
domain.bound_list[1].define_bc({'u': 0})
domain.bound_list[2].define_bc({'u': 0})
domain.bound_list[3].define_bc({'u_x': 0})
# Sample points
domain.sampling_lhs([1000, 10, 10,10])
# Training
model0 = df.PINN(input_vars=['x'], output_vars=['u'], width=32, length=6)
model1, model1_best = model0.train_adam(
    calc_loss = df.calc_loss_weighted(domain, bc_weights=1),
    learning_rate=0.001,
    epochs=2000)
model2 = model1.train_lbfgs(
    calc_loss = df.calc_loss_weighted(domain, bc_weights=1),
    epochs=200)
# Evaluate the best model
prediction = domain.bound_list[0].evaluate(model1)
prediction.sampling_line(n_points=100)
prediction.plot(y_axis='u')