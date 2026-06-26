#!/usr/bin/env python
# coding: utf-8

# # Channel Flow (Steady) DEMO code

# This notebook demonstrates solving steady-state Flow around cylinder with the same setup as this paper https://arxiv.org/abs/2002.10558

# In[1]:


import deepflow as df
print("Deepflow is runned on:", df.device) # to change to cpu use df.device = 'cpu'
df.manual_seed(69) # for reproducibility


# ## 1. Define Geometry Domain
# Set up the computational domain: a rectangle with a circular obstacle (cylinder). This defines the area for simulation.

# In[2]:


circle = df.geometry.circle(0.2, 0.2, 0.05)
rectangle = df.geometry.rectangle([0,1.1], [0,0.41])
area = rectangle - circle


# In[3]:


domain = df.domain(area, circle.bound_list)
domain.show_setup()


# ## 2. Define Physics
# Define the Navier-Stokes equations for fluid flow and apply boundary conditions (e.g., no-slip walls, inlet velocity).

# In[4]:


domain.bound_list[0].define_bc({'u': ['y', lambda x:  4*1*(0.41-x)*x/0.41**2], 'v': 0})
domain.bound_list[1].define_bc({'u': 0,'v': 0})
domain.bound_list[2].define_bc({'p': 0})
domain.bound_list[3].define_bc({'u': 0,'v': 0})
domain.bound_list[4].define_bc({'u': 0, 'v': 0})
domain.bound_list[5].define_bc({'u': 0, 'v': 0})
domain.area_list[0].define_pde(df.NavierStokes(U=1, L=1, mu=0.02, rho=1))
domain.show_setup()


# Sample initial points for training.

# In[5]:


domain.sampling_lhs(bound_sampling_res=[1000, 1000, 1000, 1000, 1000, 1000], area_sampling_res=[4000])
domain.show_coordinates(display_physics=False)


# ## 3. Train the PINN model

# Define how collocation points are sampled during training.

# In[ ]:


def do_in_adam(epoch, model):
    return

def do_in_lbfgs(epoch, model):
    if epoch % 100 == 0 and epoch > 0:
        domain.sampling_R3(bound_sampling_res=[1000, 1000, 1000, 1000, 1000, 1000], area_sampling_res=[4000])
        print(domain)


# Train the model using Adam for initial training (faster convergence).

# In[ ]:


model0 = df.PINN(width=50, length=5, input_vars=['x','y'], output_vars=['u','v','p'])

# Train the model
model1, model1best = model0.train_adam(
    learning_rate=0.004,
    epochs=2000,
    calc_loss=df.calc_loss_simple(domain),
    threshold_loss=0.01,
    do_between_epochs=do_in_adam)


# Refine the model using LBFGS for higher precision.

# In[8]:


# Train the model
model2 = model1best.train_lbfgs(
    calc_loss=df.calc_loss_simple(domain),
    epochs=450,
    threshold_loss=0.0001,
    do_between_epochs=do_in_lbfgs)


# In[9]:


domain.show_coordinates()


# Save or Load the model for later use

# In[10]:


model2.save_as_pickle("model.pkl")
model2 = df.load_from_pickle("model.pkl")


# ## 4. Visualization

# ### 4.1 Visualize area

# In[11]:


# Create object for evaluation
area_eval = domain.area_list[0].evaluate(model2)
# Sampling uniform points
area_eval.sampling_area([300, 150])
# Show available data's key
print(area_eval)


# In[21]:


area_eval.plot_color('u', s=2, cmap='rainbow').savefig("colorplot_u.png")
_ = area_eval.plot_color('v', s=2, cmap='rainbow')
_ = area_eval.plot_color('p', s=2, cmap='rainbow')
_ = area_eval.plot_streamline('u', 'v', cmap = 'jet')
_ = area_eval.plot('pde_residual')


# ### 4.2 Visualize bound

# In[16]:


# Create object for evaluation
bound_visual = domain.bound_list[2].evaluate(model2)
bound_visual.sampling_line(200) # Sampling uniform points


# In[17]:


_ = bound_visual.plot_color('u', cmap = 'rainbow')
_ = bound_visual.plot(x_axis = 'y', y_axis='u')


# ## 4.3 Visualize Neural Network data

# In[19]:


_ = bound_visual.plot_loss_curve(log_scale=True)


# ### 4.4 Export data

# In[20]:


# store the x,y,velocity_magnitude data
x_data = bound_visual.data_dict['x']
y_data = bound_visual.data_dict['y']
u_data = bound_visual.data_dict['u']

#save as txt file
import numpy as np
array = np.column_stack((x_data, y_data, u_data))
np.savetxt('outlet_velocity.txt', array)

