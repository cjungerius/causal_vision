import torch
from torch import nn, randn
from torch import Tensor as T
from torch.optim import Adam
import torch.nn.functional as F
from section_6.utils import generate_batch_of_wm_targets, generate_sensory_input_non_spatial, generate_blank_sensory_input, loss_function_non_spatial, accuracy_non_spatial
from section_chris.utils_chris import generate_same_different, generate_second_sensory_input
from tqdm import tqdm
import matplotlib.pyplot as plt

class MyRNN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dt: float, tau: float):
        super(MyRNN, self).__init__()

        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)
        self.tau = tau
        self.dt = dt
        self.alpha = self.dt / self.tau
    
     
    def init_hidden(self, batch_size):
        self.hidden = randn(batch_size, self.hidden_size)

    def step(self, input: T, noise: T, return_output: bool=False):
        leaky_term = (1 - self.alpha) * self.hidden
        
        weighted_rates = self.W(F.relu(self.hidden))
        input_projection = self.W_in(input)
        
        integration_term = self.alpha * (weighted_rates + noise + input_projection)
        
        self.hidden = leaky_term + integration_term
        
        if return_output:
            output = self.W_out(F.relu(self.hidden))
            return self.hidden, output
        else:
            return self.hidden
   
    
## Define our task
n_a = 15    # number of possible inputs

T_prestim = 0.1     # in seconds
T_stim_1 = 0.5
T_delay_1 = 0.5
T_stim_2 = 0.5
T_delay_2 = 0.0
T_resp = 0.1

## Define our system
dt = 0.01
tau = 0.2
N = 15
batch_size = 32
rnn = MyRNN(n_a + 1, N, n_a, dt, tau)

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
    rnn.init_hidden(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step(prestim_sensory_input, eta)

    stim_1_sensory_input = generate_sensory_input_non_spatial(target_indices, True, n_a)
    for ts in range(stim_1_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step(stim_1_sensory_input, eta)

    delay_1_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(delay_1_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step(delay_1_sensory_input, eta)

    #stim_2_sensory_input = generate_second_sensory_input(target_indices, same_different, True, n_a)
    stim_2_sensory_input = generate_sensory_input_non_spatial(distractor_indices, True, n_a)
    for ts in range(stim_2_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step(stim_2_sensory_input, eta)

    delay_2_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(delay_2_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step(delay_2_sensory_input, eta)

    batch_network_outputs = []

    resp_sensory_input = generate_blank_sensory_input(n_a, False, batch_size)
    for ts in range(resp_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        voltage, network_output = rnn.step(resp_sensory_input, eta, True)
        batch_network_outputs.append(network_output)

    all_network_outputs = torch.stack(batch_network_outputs, 1)
    loss = loss_function_non_spatial(target_indices, all_network_outputs)
    accs = accuracy_non_spatial(target_indices, all_network_outputs.detach())
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