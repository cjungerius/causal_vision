from section_4.feed_forward_neuron import PostSynapticNeuron

import matplotlib.pyplot as plt

import torch
from torch import randn, pi


## Define our system
dt = 0.005
tau = 0.1
N = 5
post_syn = PostSynapticNeuron(dt, N, tau)


## Define the input we are getting (see text)
frequencies = randn(N)
magnitudes = randn(N).abs()
offset = magnitudes.max()
simulation_duration = 10    # seconds
num_timesteps = int(simulation_duration / dt)


## Pregenerate the input
time_input = torch.linspace(0, simulation_duration, num_timesteps + 1).unsqueeze(-1).repeat(1, N)
input_sequence = torch.cos(2 * pi * frequencies * time_input) * magnitudes + offset

all_voltages = []
all_weighted_inputs = []


## Loop through and get the voltage simulation
for ts in range(num_timesteps):

    # Get input rates for that timestep
    input_rates = input_sequence[ts]

    # Feed into the output neuron and collect new voltage
    v_m_t, w_times_r = post_syn.step_dynamics(input_rates)
    all_voltages.append(v_m_t.item())
    all_weighted_inputs.append(w_times_r.item())


# Plot voltage over time
plt.plot(time_input[1:,0].numpy(), all_voltages, label = '$v_m$')
plt.plot(time_input[1:,0].numpy(), all_weighted_inputs, label = '$w\cdot r$')
plt.legend()
plt.show()

