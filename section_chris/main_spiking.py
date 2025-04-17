from section_chris.feed_forward_spiking_neuron import PostSynapticSpikingNeuron

import matplotlib.pyplot as plt

import torch
from torch import randn, pi


## Define our system
dt = 0.001
tau = 0.3
N = 1
theta = -.3
v_rest = -.7
post_syn = PostSynapticSpikingNeuron(dt, N, tau, theta, v_rest)


## Define the input we are getting (see text)
frequencies = randn(N)
magnitudes = randn(N).abs()
offset = magnitudes.max()
simulation_duration = 10    # seconds
num_timesteps = int(simulation_duration / dt)


## Pregenerate the input
time_input = torch.linspace(0, simulation_duration, num_timesteps + 1).unsqueeze(-1).repeat(1, N)
#input_sequence = torch.cos(2 * pi * frequencies * time_input) * magnitudes + offset
input_sequence = torch.cos(2 * pi * frequencies * time_input) * magnitudes + offset

all_spikes = []
all_weighted_inputs = []


## Loop through and get the voltage simulation
for ts in range(num_timesteps):

    # Get input rates for that timestep
    input_rates = input_sequence[ts]

    # Feed into the output neuron and collect new voltage
    spike_t, w_times_r = post_syn.step_dynamics(input_rates)
    all_spikes.append(spike_t.item())
    all_weighted_inputs.append(w_times_r.item())


# Plot voltage over time
plt.plot(time_input[1:,0].numpy(), all_spikes, label = r'$spikes$')
plt.plot(time_input[1:,0].numpy(), all_weighted_inputs, label = r'$w \cdot r$')
plt.legend()
plt.show()

