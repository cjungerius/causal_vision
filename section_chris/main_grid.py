from section_6.rate_rnn_with_io import ReluRateRNNWithIO
from section_6.utils import generate_blank_sensory_input
from section_chris.utils_chris import generate_batch_of_2d_wm_targets, generate_2d_sensory_input, loss_function_grid_2, errors_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch import randn

from sklearn.decomposition import PCA


## Define our task
n_a = 5    # size of sides of square


## Define our system
dt = 0.01
tau = 0.1
N = 30
batch_size = 32
rnn = ReluRateRNNWithIO(dt, N, n_a**2 + 1, 2, tau)    # Refer to notes to understand why the i/o are these sizes!


## Define training machinery
lr = 1e-3
opt = Adam(rnn.parameters(), lr)
num_batches = 4000
losses = [] # Store for loss curve plotting
x_errs = [] # Store for accuracies curve plotting
y_errs = []

## Define simulation parameters
T_prestim = 0.2     # in seconds
T_stim = 0.5
T_delay = 0
T_resp = 0.1

prestim_timesteps = int(T_prestim / dt)
stim_timesteps = int(T_stim / dt)
delay_timesteps = int(T_delay / dt)
resp_timesteps = int(T_resp / dt)

eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = 0.1


## Initialise simulation, including the first noise term
eta_tilde = randn(batch_size, N)


### Begin training
for b in tqdm(range(num_batches)):

    opt.zero_grad()

    target_indices = generate_batch_of_2d_wm_targets(n_a, batch_size)
    rnn.initialise_u(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a**2, True, batch_size)
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(prestim_sensory_input, eta, False)

    stim_sensory_input = generate_2d_sensory_input(target_indices, True, n_a)
    for ts in range(stim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_sensory_input, eta, False)

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
    loss = loss_function_grid_2(target_indices, all_network_outputs, n_a)
    x_err, y_err = errors_grid(target_indices, all_network_outputs.detach(), n_a)
    loss.backward()
    opt.step()

    losses.append(loss.item())
    x_errs.append(x_err.item())
    y_errs.append(y_err.item())

    if b % 200 == 0:
        plt.close('all')
        fig, axes = plt.subplots(3)
        
        axes[0].plot(losses)
        axes[1].plot(x_errs)
        axes[2].plot(y_errs)

        print(losses[-10:])
        print(x_errs[-10:])
        print(y_errs[-10:])
        fig.savefig('section_chris/grid_losses.png')


### RUN TEST TRIAL FOR ACTIVITY
target_indices =  torch.tensor([[i,j] for i in range(1,n_a+1) for j in range(1,n_a+1)])
test_batch_size = target_indices.shape[0]
rnn.initialise_u(test_batch_size)
eta_tilde = randn(test_batch_size, N)

prestim_sensory_input = generate_blank_sensory_input(n_a**2, True, test_batch_size)
for ts in range(prestim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(prestim_sensory_input, eta, False)

stim_sensory_input = generate_2d_sensory_input(target_indices, True, n_a)
for ts in range(stim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(stim_sensory_input, eta, False)

delay_sensory_input = generate_blank_sensory_input(n_a**2, True, test_batch_size)
for ts in range(delay_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(delay_sensory_input, eta, False)

test_trial_voltages = []

resp_sensory_input = generate_blank_sensory_input(n_a**2, False, test_batch_size)
for ts in range(resp_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
    test_trial_voltages.append(voltage)

all_test_trial_voltages = torch.stack(test_trial_voltages, 1).detach().numpy()
print(all_test_trial_voltages)
dim_reduced_voltages_flat = PCA(n_components=2).fit_transform(all_test_trial_voltages.reshape(test_batch_size * resp_timesteps, -1))
dim_reduced_voltages = dim_reduced_voltages_flat.reshape(test_batch_size, resp_timesteps, -1)

fig, axes = plt.subplots(1)
for i, drv in enumerate(dim_reduced_voltages):
    axes.plot(*drv.T, label = target_indices[i], color=plt.cm.RdYlBu(i/(n_a**2)))
    axes.scatter(*drv[0,:,None], marker='o', color=plt.cm.RdYlBu(i/(n_a**2)))
    axes.scatter(*drv[-1,:,None], marker='x', color=plt.cm.RdYlBu(i/(n_a**2)))
axes.legend()
fig.suptitle('PCA of trajectories. Response period starts at o and ends at x')
fig.savefig('section_chris/grid_pca_reprs.png')
