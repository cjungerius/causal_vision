from section_6.rate_rnn_with_io import ReluRateRNNWithIO
from section_6.utils import generate_blank_sensory_input
from section_chris.utils_chris import generate_batch_of_2d_wm_targets, generate_sensory_input_2d_vonmises, loss_function_grid_2, errors_spatial_2d, torus_embedding
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch import randn

from sklearn.decomposition import PCA


## Define our task
n_a = 4    # points along each circular dimension
A = 1.0
kappa = 3.0


## Define our system
dt = 0.01
tau = 0.1
N = 30
batch_size = 32
rnn = ReluRateRNNWithIO(dt, N, n_a**2 + 1, 4, tau)    # Refer to notes to understand why the i/o are these sizes!


## Define training machinery
lr = 1e-3
opt = Adam(rnn.parameters(), lr)
num_batches = 1000
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

    stim_sensory_input = generate_sensory_input_2d_vonmises(target_indices, True, n_a, A, kappa)
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
    x_err, y_err = errors_spatial_2d(target_indices, all_network_outputs.detach(), n_a)
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

test_trial_voltages = []

resp_sensory_input = generate_blank_sensory_input(n_a**2, False, test_batch_size)
for ts in range(resp_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
    test_trial_voltages.append(voltage)

all_test_trial_voltages = torch.stack(test_trial_voltages, 1).detach()

mean_voltages = all_test_trial_voltages.mean(axis=1)

theta1 = torch.atan2(mean_voltages[:, 1], mean_voltages[:, 0])  # angle 1
theta2 = torch.atan2(mean_voltages[:, 3], mean_voltages[:, 2])  # angle 2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
points = torus_embedding(theta1, theta2)
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
plt.show()