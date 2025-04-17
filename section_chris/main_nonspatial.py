from section_chris.utils_chris import generate_same_different, generate_second_sensory_input
from section_6.rate_rnn_with_io import ReluRateRNNWithIO
from section_6.utils import generate_batch_of_wm_targets, generate_sensory_input_non_spatial, generate_blank_sensory_input, loss_function_non_spatial, accuracy_non_spatial

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch import randn

from sklearn.decomposition import PCA

## Define our task
n_a = 40    # number of possible inputs

T_prestim = 0.1     # in seconds
T_stim_1 = 0.5
T_delay_1 = 0.5
T_stim_2 = 0.5
T_delay_2 = 0.0
T_resp = 0.1

## Define our system
dt = 0.01
tau = 0.2
N = 60
batch_size = 48
rnn = ReluRateRNNWithIO(dt, N, n_a + 1, 2, tau)    # Refer to notes to understand why the i/o are these sizes!

## Define training machinery
lr = 1e-3
opt = Adam(rnn.parameters(), lr)
num_batches = 4000
losses = [] # Store for loss curve plotting
accuracies = [] # Store for accuracies curve plotting

## Define simulation parameters
prestim_timesteps = int(T_prestim / dt)
stim_1_timesteps = int(T_stim_1 / dt)
delay_1_timesteps = int(T_delay_1 / dt)
stim_2_timesteps = int(T_stim_2 / dt)
delay_2_timesteps = int(T_delay_2 / dt)
resp_timesteps = int(T_resp / dt)

eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = 0.1

## Initialise simulation, including the first noise term
eta_tilde = randn(batch_size, N)

### Begin training
for b in tqdm(range(num_batches)):

    opt.zero_grad()
    
    target_indices = generate_batch_of_wm_targets(n_a, batch_size)
    distractor_indices = generate_batch_of_wm_targets(n_a, batch_size)
    same_different = generate_same_different(batch_size)
    rnn.initialise_u(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(prestim_sensory_input, eta, False)

    stim_1_sensory_input = generate_sensory_input_non_spatial(target_indices, True, n_a)
    for ts in range(stim_1_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_1_sensory_input, eta, False)

    delay_1_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(delay_1_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_1_sensory_input, eta, False)

    stim_2_sensory_input = generate_second_sensory_input(target_indices, same_different, True, n_a)
    for ts in range(stim_2_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_2_sensory_input, eta, False)

    delay_2_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(delay_2_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_2_sensory_input, eta, False)

    batch_network_outputs = []

    resp_sensory_input = generate_blank_sensory_input(n_a, False, batch_size)
    for ts in range(resp_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
        batch_network_outputs.append(network_output)

    all_network_outputs = torch.stack(batch_network_outputs, 1)
    loss = loss_function_non_spatial(same_different, all_network_outputs)
    accs = accuracy_non_spatial(same_different, all_network_outputs.detach())
    loss.backward()
    opt.step()#    print(batch_network_outputs)

    losses.append(loss.item())
    accuracies.append(accs.item())

    if b % 200 == 0:
        plt.close('all')
        fig, axes = plt.subplots(2)
        axes[0].plot(losses)
        axes[1].plot(accuracies)

        print(losses[-10:])
        plt.savefig('losses.png')