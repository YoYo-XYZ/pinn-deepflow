from deepflow import PINN, Geometry, HardConstraint
import torch

line0 = Geometry.line_horizontal(0,[0,1])
line0.define_bc({'u': HardConstraint(1), 'p': HardConstraint(1)})

line1 = Geometry.line_horizontal(1, [0, 1])
line1.define_bc({'u': HardConstraint(1), 'p': HardConstraint(1)})

line_test = Geometry.line_horizontal(0, [0,1])
X,Y = line_test.sampling_line(10)
input_dict = {'x':X,'y':Y}

model = PINN(10,10)
model.apply_hard_constraints([line0, line1])
output = model(input_dict)
print(output['v'])
