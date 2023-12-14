from section_5.rate_rnn import ReluRateRNN

import matplotlib.pyplot as plt

import torch
from torch import randn, pi


## Define our system
dt = 0.005
tau = 0.1
N = 2
rnn = ReluRateRNN(dt, N, tau)
rnn.eval()


## Define simulation parameters
simulation_duration = 10    # seconds
num_timesteps = int(simulation_duration / dt)
eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = 0.1


## Initialise simulation, including the first noise term
eta_tilde = randn(N)
all_voltages = []


## Loop through and get the voltage simulation
for ts in range(num_timesteps):

    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(N))
    eta = eta_tilde * C

    new_voltage = rnn.step_dynamics(eta)    # rnn voltages are updated internally!
    all_voltages.append(new_voltage)

# Stack to make things easier for ourselves
all_voltages = torch.stack(all_voltages, 0)     # [num timesteps, N = 2]
all_rates = rnn.non_linearity(all_voltages)

# Plot voltage over time
plt.plot(all_voltages[:,0].detach().numpy(), all_voltages[:,1].detach().numpy(), label = '$u(t)$')
plt.plot(all_rates[:,0].detach().numpy(), all_rates[:,1].detach().numpy(), label = '$r(t)$')
plt.legend()
plt.show()

