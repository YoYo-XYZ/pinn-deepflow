import os  # Uncomment this line to disable GPU
os.environ["DDE_BACKEND"] = "pytorch"  # Must be before importing deepxde
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch

print("Backend: pytorch")
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

# Verify DeepXDE is using GPU
try:
    x = dde.backend.as_tensor([1.0])
    print(f"DeepXDE tensor device: {x.device}")
except Exception as e:
    print(f"Could not check DeepXDE tensor device directly: {e}")

rho = 1000
mu = 0.001
u_in = 0.0001
D = 1
L = 2

geom = dde.geometry.Rectangle(xmin=[-L/2,-D/2], xmax=[L/2,D/2])                       # Rectangle xmin = the minimum point in for both the direction. xmax is the maximum for both direction.

def boundary_wall(X, on_boundary):                                                    # This function takesin points X = (x,y) and whether it is on boundary or not (on_boundary is a dde built-in. When the random points are generated, dde mark the boundary points.).
    #print("X", X)                                                                    # So, we know if they are on boundaries, but not which one. We don't want the points on inlet and outlet boundaries. Only on the wall boundaries.
    #print("on_boundary", on_boundary)
    
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], -D/2, rtol=1e-05, atol=1e-08), np.isclose(X[1], D/2, rtol=1e-05, atol=1e-08)), on_boundary)
    
    '''
    Logical function gives either true or false, on_wall = 1/0, 1= on wall boundary, 0 = not (Maybe a Boundary point but not on wall).
    
    np.logical_and is a (&& - and kind) of condition (both argument should satisfy), np.logical_or is a (|| - or kind) of condition (Either of them satisfy then true).
    In a and argument ( or argument (either on top wall or bottom wall = 1), points on boundary = 1), if those both are 1 then on_wall = 1.
    
    np.isclose(a,b) gives true/false, true-> if a is close to b in the tolerance given. Here in this problem, X=(x,y) so X[1] is the value on y axis. We dont care for x value, just figuring if X[1] is close to -D/2 or D/2, then it is on wall boundary.
    '''
    return on_wall

def boundary_inlet(X, on_boundary):
    on_inlet = np.logical_and(np.isclose(X[0], -L/2, rtol=1e-05, atol=1e-08), on_boundary)
    '''
    Logical function gives either true or false, on_inlet = 1/0, 1= on inlet boundary, 0 = not (Maybe a Boundary point but not on inlet boundary).
    
    np.logical_and is a (&& - and kind) of condition (both argument should satisfy), np.logical_or is a (|| - or kind) of condition (Either of them satisfy then true).
    In a and argument ( on inlet boundary = 1), points on boundary = 1), if those both are 1 then on_inlet = 1.
    
    np.isclose(a,b) gives true/false, true-> if a is close to b in the tolerance given. Here in this problem, X=(x,y) so X[0] is the value on x axis. We dont care for y value, just figuring if X[0] is close to -L/2, then it is on inlet boundary.
    '''
    return on_inlet

def boundary_outlet(X, on_boundary):
    on_outlet = np.logical_and(np.isclose(X[0], L/2, rtol=1e-05, atol=1e-08), on_boundary)  # Same logic as above
    return on_outlet

bc_wall_u = dde.DirichletBC(geom, lambda X:0., boundary_wall, component= 0)           # Component = 0 meaning out of the unknowns (u,v,p), it is u. Lambda function says that the u is 0 (No slip) on boundary walls, which (boundary walls) is previously defined by a function.
bc_wall_v = dde.DirichletBC(geom, lambda X:0., boundary_wall, component= 1)           # Component = 1 meaning out of the unknowns (u,v,p), it is v. Lambda function says that the v is 0 (No slip) on boundary walls, which (boundary walls) is previously defined by a function.

bc_inlet_u = dde.DirichletBC(geom, lambda X:u_in, boundary_inlet, component= 0)       # Component = 0 meaning out of the unknowns (u,v,p), it is u. Lambda function says that the u is u_inlet on inlet wall, which (inlet wall) is previously defined by a function.
bc_inlet_v = dde.DirichletBC(geom, lambda X:0., boundary_inlet, component= 1)         # Component = 1 meaning out of the unknowns (u,v,p), it is v. Lambda function says that the v is 0 on inlet wall, which (inlet wall) is previously defined by a function.

bc_outlet_v = dde.DirichletBC(geom, lambda X:0., boundary_outlet, component= 1)       # Component = 1 meaning out of the unknowns (u,v,p), it is v. Lambda function says that the v is 0 on outlet wall, which (outlet wall) is previously defined by a function.
bc_outlet_p = dde.DirichletBC(geom, lambda X:0., boundary_outlet, component= 2)       # Component = 2 meaning out of the unknowns (u,v,p), it is p. Lambda function says that the p is 0 on outlet wall, which (outlet wall) is previously defined by a function.

def pde(X, Y):                                                                        # X = (x,y), Y = (u,v,p)
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)                                          # Jacobian matrix's element is grabbed. i & j value is used to tell which element we need.
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)
    
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)
    
    du_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)                             # Components = (u,v,p). component = 0 = u
    du_yy = dde.grad.hessian(Y, X, component=0, i=1, j=1)                             # Components = (u,v,p). component = 0 = u
    
    dv_xx = dde.grad.hessian(Y, X, component=1, i=0, j=0)                             # Components = (u,v,p). component = 1 = v
    dv_yy = dde.grad.hessian(Y, X, component=1, i=1, j=1)                             # Components = (u,v,p). component = 1 = v
    
    # Momentum Equations (u direction & v direction, it is a NS steady state eqs.-> Can be find in written notes).
    # Y = [u, v, p]. u = Y[:,0:1] (0:1 ensures the shape of u -> (n,) instead of (n,1))
    # Same way v = Y[:,1:2], and p = Y[:,2:]
    pde_u = Y[:,0:1] * du_x + Y[:,1:2] * du_y + 1/rho * dp_x - (mu/rho) * (du_xx + du_yy)
    pde_v = Y[:,0:1] * dv_x + Y[:,1:2] * dv_y + 1/rho * dp_y - (mu/rho) * (dv_xx + dv_yy)
    
    # Continuity Eq.
    pde_cont = du_x + dv_y
    
    return [pde_u, pde_v, pde_cont]


data = dde.data.PDE(
    geom,
    pde,
    [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v],
    num_domain= 2000,
    num_boundary= 200,
    num_test= 1000
)

plt.figure(figsize=(10,5))
plt.scatter(data.train_x_all[:,0], data.train_x_all[:,1], s=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Defining the model
net = dde.maps.FNN([2] + [40]*8 + [3], "tanh", "Glorot uniform")                     # Network starts with 2 neurons -> x,y. 5 hidden layers with 64 neurons. Last layer with 3 neurons -> u,v,p.

model = dde.Model(data, net)

# Training with Adam first
model.compile("adam", lr = 1e-3)

checker = dde.callbacks.ModelCheckpoint(
    "model.ckpt", save_better_only=True, period=100
)

loss_history, train_state = model.train(epochs=1000, callbacks=[checker], display_every=1)


# Training with L-BFGS for convergence
# Model saving option is not available for L-BFGS optimizer
dde.optimizers.config.set_LBFGS_options(maxiter=0)

model.compile("L-BFGS")
loss_history, train_state = model.train(display_every=1)

# To see the training and testing loss
dde.saveplot(loss_history, train_state, issave=True, isplot=True)


# Predict through the trained model and visulize.

samples = geom.random_points(50000)                                              # Geom is defined, a rectangular and its bounds. The 500000 random points will be generated in that domain (x,y) of size -> (500000,2)
results = model.predict(samples)                                                  # Results size -> (500000,3), columns are u,v,p.


color_legend = [[0,1.5], [-0.3,0.3], [0,35]]

for idx in range(3):
    plt.figure(figsize=(20,4))
    plt.scatter(samples[:,0],                                                     # x
                samples[:,1],                                                     # y
                c = results[:,idx],                                               # 3 plots for u,v,p
                cmap='jet',
                s=2)
    plt.colorbar()
    plt.clim(color_legend[idx])
    plt.xlim((0-L/2, L-L/2))
    plt.ylim((0-D/2, D-D/2))
    plt.tight_layout()
    plt.show()
