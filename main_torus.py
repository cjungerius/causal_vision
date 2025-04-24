#!/usr/bin/env python
# coding: utf-8

# In[1]:


from section_6.rate_rnn_with_io import ReluRateRNNWithIO
from section_6.utils import generate_blank_sensory_input
from section_chris.utils_chris import generate_batch_of_2d_wm_targets, generate_sensory_input_2d_vonmises, loss_function_2d_vonmises, errors_spatial_2d, generate_same_different
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch import randn, pi
from torch.distributions import VonMises

from sklearn.decomposition import PCA
%matplotlib ipympl



# In[2]:


## Define our task
n_a = 20    # points along each circular dimension
A = 1.0
kappa = 3.0


# In[53]:


## Define our system
dt = 0.01
tau = 0.1
N = 70
batch_size = 32
rnn = ReluRateRNNWithIO(dt, N, n_a**2 + 1, 4, tau)    # Refer to notes to understand why the i/o are these sizes!


# In[54]:


## Define training machinery
lr = 1e-3
opt = Adam(rnn.parameters(), lr)
num_batches = 6000
losses = [] # Store for loss curve plotting
dim1_errs = [] # Store for accuracies curve plotting
dim2_errs = []


# In[55]:


## Define simulation parameters
T_prestim = 0.2     # in seconds
T_stim = 0.5
T_delay = 0.2
T_resp = 0.1

prestim_timesteps = int(T_prestim / dt)
stim_timesteps = int(T_stim / dt)
delay_timesteps = int(T_delay / dt)
resp_timesteps = int(T_resp / dt)

eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = 0.1


# In[56]:


## Initialise simulation, including the first noise term
eta_tilde = randn(batch_size, N)


# In[57]:


### Begin training
for b in tqdm(range(num_batches)):

    opt.zero_grad()

    target_indices = generate_batch_of_2d_wm_targets(n_a, batch_size)
    distractor_indices = generate_batch_of_2d_wm_targets(n_a, batch_size)
    same_different = generate_same_different(batch_size).unsqueeze(-1).expand(-1,2)
    second_stim = torch.where(same_different==1,target_indices, distractor_indices)

    rnn.initialise_u(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a**2, True, batch_size)
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(prestim_sensory_input, eta, False)

    stim_sensory_input = generate_sensory_input_2d_vonmises(target_indices, True, n_a, A, kappa, 0.0)
    for ts in range(stim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_sensory_input, eta, False)

    # delay_sensory_input = generate_blank_sensory_input(n_a**2, True, batch_size)
    # for ts in range(delay_timesteps):
    #     eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
    #     eta = eta_tilde * C
    #     rnn.step_dynamics(delay_sensory_input, eta, False)

    # stim_sensory_input = generate_sensory_input_2d_vonmises(second_stim, True, n_a, A, kappa, 0.1)
    # for ts in range(stim_timesteps):
    #     eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
    #     eta = eta_tilde * C
    #     rnn.step_dynamics(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a**2, True, batch_size)
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_sensory_input, eta, False)

    batch_network_outputs = []

    resp_sensory_input = generate_blank_sensory_input(n_a**2, False, batch_size)
    for ts in range(resp_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
        batch_network_outputs.append(network_output)

    all_network_outputs = torch.stack(batch_network_outputs, 1)
    loss = loss_function_2d_vonmises(target_indices, all_network_outputs, n_a)
    x_err, y_err = errors_spatial_2d(target_indices, all_network_outputs.detach(), n_a)
    loss.backward()
    opt.step()

    losses.append(loss.item())
    dim1_errs.append(x_err.item())
    dim2_errs.append(y_err.item())

    if b % 200 == 0:
        plt.close('all')
        fig, axes = plt.subplots(3)

        axes[0].plot(losses[50:])
        axes[1].plot(dim1_errs[50:])
        axes[2].plot(dim2_errs[50:])

        print(losses[-10:])
        fig.savefig('section_chris/grid_losses.png')


# In[58]:


### RUN TEST TRIAL FOR ACTIVITY
#target_indices =  torch.tensor([[i,j] for i in range(1,n_a+1) for j in range(1,n_a+1)])
target_indices = torch.randn(20,2)
test_batch_size = target_indices.shape[0]
rnn.initialise_u(test_batch_size)
eta_tilde = randn(test_batch_size, N)

prestim_sensory_input = generate_blank_sensory_input(n_a**2, True, test_batch_size)
for ts in range(prestim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(prestim_sensory_input, eta, False)

stim_sensory_input = generate_sensory_input_2d_vonmises(target_indices, True, n_a, A, kappa)
for ts in range(stim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(stim_sensory_input, eta, False)

delay_sensory_input = generate_blank_sensory_input(n_a**2, True, test_batch_size)
for ts in range(delay_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(delay_sensory_input, eta, False)

# stim_sensory_input = generate_sensory_input_2d_vonmises(target_indices, True, n_a, A, kappa)
# for ts in range(stim_timesteps):
#     eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
#     eta = eta_tilde * C
#     rnn.step_dynamics(stim_sensory_input, eta, False)

# delay_sensory_input = generate_blank_sensory_input(n_a**2, True, test_batch_size)
# for ts in range(delay_timesteps):
#     eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
#     eta = eta_tilde * C
#     rnn.step_dynamics(delay_sensory_input, eta, False)

test_trial_voltages = []
test_trial_outputs = []

resp_sensory_input = generate_blank_sensory_input(n_a**2, False, test_batch_size)
for ts in range(resp_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
    test_trial_voltages.append(voltage)
    test_trial_outputs.append(network_output)

all_test_trial_voltages = torch.stack(test_trial_voltages, 1).detach()


# In[59]:


y_hat = torch.stack(test_trial_outputs,1).detach().mean(1)
y = target_indices * 2 * torch.pi / n_a


# In[74]:


def plot_outputs_on_torus(outputs, targets, R=2.0, r=.5, z_scale=.4):
    x1, y1 = outputs[:, 0], outputs[:, 1]
    x2, y2 = outputs[:, 2], outputs[:, 3]

    # Convert outputs to angles
    theta1_hat = torch.atan2(y1, x1)
    theta2_hat = torch.atan2(y2, x2)

    theta1 = targets[:,0]
    theta2 = targets[:,1]

    # Convert to torus coordinates
    X_hat = (R + r * torch.cos(theta2_hat)) * torch.cos(theta1_hat)
    Y_hat = (R + r * torch.cos(theta2_hat)) * torch.sin(theta1_hat)
    Z_hat = z_scale * r * torch.sin(theta2_hat)

    X = (R + r * torch.cos(theta2)) * torch.cos(theta1)
    Y = (R + r * torch.cos(theta2)) * torch.sin(theta1)
    Z = z_scale * r * torch.sin(theta2)

    # Create torus surface (grid)
    n_grid = 50
    t1 = np.linspace(0, 2 * np.pi, n_grid)
    t2 = np.linspace(0, 2 * np.pi, n_grid)
    T1, T2 = np.meshgrid(t1, t2)
    X_surf = (R + r * np.cos(T2)) * np.cos(T1)
    Y_surf = (R + r * np.cos(T2)) * np.sin(T1)
    Z_surf = r * z_scale * np.sin(T2)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the torus surface
    ax.plot_surface(X_surf, Y_surf, Z_surf, rstride=1, cstride=1, color='lightgray', alpha=0.3, edgecolor='none')

    # Plot the output points
    ax.scatter(X.numpy(), Y.numpy(), Z.numpy(), color='blue', s=20, alpha=0.8)
    ax.scatter(X_hat.numpy(), Y_hat.numpy(), Z_hat.numpy(), color='red', s=20, alpha=0.8)

    for i in range(X.size(0)):
        ax.plot([X[i], X_hat[i]], [Y[i],Y_hat[i]],zs=[Z[i],Z_hat[i]], color="black")

    ax.set_zlim3d(-.5,.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Network Outputs on 3D Torus Surface")
    ax.view_init(elev=30, azim=45)  # Nice viewing angle
    plt.tight_layout()
    plt.show()


# In[75]:


plot_outputs_on_torus(y_hat, y)







