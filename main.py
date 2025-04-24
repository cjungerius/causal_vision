#!/usr/bin/env python
# coding: utf-8

# In[1]:

# %matplotlib widget
from section_6.rate_rnn_with_io import ReluRateRNNWithIO
from section_6.utils import generate_blank_sensory_input, generate_sensory_input_spatial, loss_function_spatial, errors_spatial
from section_chris.utils_chris import generate_batch_of_continuous_wm_targets, generate_sensory_input_2d_vonmises, loss_function_2d_vonmises, errors_spatial_2d, generate_same_different
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




# In[2]:


## Define our task
n_a = 10    # points along circular dimension
A = 1.0
kappa = 3.0


# In[53]:


## Define our system
dt = 0.01
tau = 0.2
N = 50
batch_size = 32
rnn = ReluRateRNNWithIO(dt, N, n_a + 1, 2, tau)    # Refer to notes to understand why the i/o are these sizes!


# In[54]:


## Define training machinery
lr = 1e-3
opt = Adam(rnn.parameters(), lr)
num_batches = 4000
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

    target_indices = generate_batch_of_continuous_wm_targets(n_a, batch_size)
    distractor_indices = generate_batch_of_continuous_wm_targets(n_a, batch_size)
    same_different = generate_same_different(batch_size)
    second_stim = torch.where(same_different==1,target_indices, distractor_indices)

    rnn.initialise_u(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(prestim_sensory_input, eta, False)

    stim_sensory_input = generate_sensory_input_spatial(target_indices, True, n_a, A, kappa)
    for ts in range(stim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_sensory_input, eta, False)

    stim_sensory_input = generate_sensory_input_spatial(second_stim, True, n_a, A, kappa)
    for ts in range(stim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_sensory_input, eta, False)

    batch_network_outputs = []

    resp_sensory_input = generate_blank_sensory_input(n_a, False, batch_size)
    for ts in range(resp_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
        batch_network_outputs.append(network_output)

    all_network_outputs = torch.stack(batch_network_outputs, 1)
    loss = loss_function_spatial(target_indices, all_network_outputs, n_a)
    x_err, y_err = errors_spatial(target_indices, all_network_outputs.detach(), n_a)
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
target_indices = torch.rand(20,) * n_a
distractor_indices = torch.rand(20,) * n_a
test_batch_size = target_indices.shape[0]
rnn.initialise_u(test_batch_size)
eta_tilde = randn(test_batch_size, N)
same_different = generate_same_different(test_batch_size)
second_stim = torch.where(same_different==1,target_indices, distractor_indices)


prestim_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)
for ts in range(prestim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(prestim_sensory_input, eta, False)

stim_sensory_input = generate_sensory_input_spatial(target_indices, True, n_a, A, kappa)
for ts in range(stim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(stim_sensory_input, eta, False)

delay_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)
for ts in range(delay_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(delay_sensory_input, eta, False)

stim_sensory_input = generate_sensory_input_spatial(second_stim, True, n_a, A, kappa)
for ts in range(stim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(stim_sensory_input, eta, False)

delay_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)
for ts in range(delay_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(delay_sensory_input, eta, False)

test_trial_voltages = []
test_trial_outputs = []

resp_sensory_input = generate_blank_sensory_input(n_a, False, test_batch_size)
for ts in range(resp_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
    test_trial_voltages.append(voltage)
    test_trial_outputs.append(network_output)

all_test_trial_voltages = torch.stack(test_trial_voltages, 1).detach()


# In[59]:


y_hat = torch.stack(test_trial_outputs,1).detach().mean(1)
y_hat = torch.atan2(y_hat[:,1],y_hat[:,0])
y = target_indices * 2 * torch.pi / n_a




# %%
