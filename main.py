# coding: utf-8

# In[1]:

# %matplotlib widget
from section_6.rate_rnn_with_io import ReluRateRNNWithIO
from section_6.utils import generate_blank_sensory_input, generate_sensory_input_spatial, loss_function_spatial, errors_spatial, loss_function_non_spatial
from section_chris.utils_chris import generate_batch_of_continuous_wm_targets, generate_sensory_input_2d_vonmises, loss_function_2d_vonmises, errors_spatial_2d, generate_same_different, loss_function_cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch.optim import Adam
from torch import randn, pi
from torch.distributions import VonMises
import seaborn as sns
import pandas as pd
from scipy.special import iv as besselI
from scipy.optimize import minimize_scalar
from scipy.stats import vonmises
from itertools import product

from sklearn.decomposition import PCA




# In[2]:


## Define our task
n_a = 100000   # points along circular dimension
A = 1.0
kappa = 3

# In[53]:


## Define our system
dt = 0.01
tau = 0.25  # time constant for the RNN dynamics
N = 200 
batch_size = 200
rnn = ReluRateRNNWithIO(dt, N, n_a, 2, tau)    # Refer to notes to understand why the i/o are these sizes!


# In[54]:


## Define training machinery
lr = 5e-4  # Reduced learning rate for larger network
opt = Adam(rnn.parameters(), lr)
num_batches = 2000
losses = [] # Store for loss curve plotting
dim1_errs = [] # Store for accuracies curve plotting
dim2_errs = []


# In[55]:


## Define simulation parameters
T_prestim = 0.1     # in seconds
T_stim_1 = 0.25
T_stim_2 = 0.25
T_delay = 0.0
T_resp = 0.1

prestim_timesteps = int(T_prestim / dt)
stim_timesteps_1 = int(T_stim_1 / dt)
stim_timesteps_2 = int(T_stim_2 / dt)
delay_timesteps = int(T_delay / dt)
resp_timesteps = int(T_resp / dt)

kappa_tilde = 7 # von Mises concentration parameter for the sensory input
# noise generating von mises distribution
noise_dist = VonMises(0, kappa_tilde)
p = 2/5

eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = .01

steps = torch.tensor([20, 35, 75, -20, -35, -75]) / 360 * n_a

## Initialise simulation, including the first noise term
eta_tilde = randn(batch_size, N)

# Ideal observer functions for training comparison
def inv_logit(x):
    return 1 / (1 + np.exp(-x))

def estimate_shared_likelihood(x1, x2, kappa):
    delta_x = np.abs(x1 - x2)
    delta_x = np.where(delta_x > np.pi, 2*np.pi - delta_x, delta_x)
    kappa_eff = 2 * kappa * np.cos(delta_x / 2)
    marginal_likelihood = besselI(0, kappa_eff) / ((2 * np.pi)**2 * besselI(0, kappa)**2)
    return {"lik_shared": marginal_likelihood, "kappa_eff": kappa_eff}

def compute_p_shared(x1, x2, kappa, p=0.5):
    lik_estimate = estimate_shared_likelihood(x1, x2, kappa)
    lik_shared = lik_estimate["lik_shared"]
    lik_indep = (1 / (2 * np.pi))**2

    log_p1 = np.log(p) + np.log(lik_shared)
    log_p2 = np.log(1 - p) + np.log(lik_indep)

    posterior_prob_shared = inv_logit(log_p1 - log_p2)
    return posterior_prob_shared

def calc_circular_mean_stable(theta, kappa_1, kappa_2, mu_1, mu_2):
    # Compute characteristic functions
    r1 = besselI(1, kappa_1) / besselI(0, kappa_1)  # Mean resultant length
    r2 = besselI(1, kappa_2) / besselI(0, kappa_2)
    
    # Complex representation
    z1 = r1 * np.exp(1j * mu_1)
    z2 = r2 * np.exp(1j * mu_2)
    
    # Mixture characteristic function
    z_mix = theta * z1 + (1 - theta) * z2
    
    # Extract circular mean
    return np.angle(z_mix)

def compute_circular_mean(angle1, angle2):
    """Compute circular mean of two angles"""
    # Convert to unit vectors and average
    x = np.cos(angle1) + np.cos(angle2)
    y = np.sin(angle1) + np.sin(angle2)
    return np.arctan2(y, x)

def compute_ideal_observer_estimates(first_inputs, second_inputs, kappa_tilde, p_prior):
    """Compute ideal observer estimates for a batch of inputs"""
    estimates = []
    
    # Convert to numpy for processing (inputs are already in the correct scale)
    first_np = first_inputs.numpy() * 2 * np.pi / n_a
    second_np = second_inputs.numpy() * 2 * np.pi / n_a
    
    for i in range(len(first_np)):
        p_shared = compute_p_shared(first_np[i], second_np[i], kappa_tilde, p_prior)
        # Correctly compute circular mean for shared cause
        mu_shared = compute_circular_mean(first_np[i], second_np[i])
        mu_shared = mu_shared % (2 * np.pi)
        
        delta_x = abs(first_np[i] - second_np[i])
        delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
        kappa_eff = 2 * kappa_tilde * np.cos(delta_x / 2)
        
        circ_mean = calc_circular_mean_stable(p_shared, kappa_eff, kappa_tilde, mu_shared, second_np[i])
        circ_mean = circ_mean % (2 * np.pi)
        # Convert back to n_a scale
        estimates.append(circ_mean * n_a / (2 * np.pi))
    
    return torch.tensor(estimates)

network_losses = []  # Store for network loss
optimal_losses = []  # Store for optimal loss

# In[57]:


### Begin training
for b in tqdm(range(num_batches)):

    opt.zero_grad()

    first_indices = generate_batch_of_continuous_wm_targets(n_a, batch_size)
    second_indices = generate_batch_of_continuous_wm_targets(n_a, batch_size)
    same_different = generate_same_different(batch_size, p)
    second_indices = torch.where(same_different==1,first_indices, second_indices)

    #batch_steps = torch.randint(6,first_indices.size())

    #second_indices = (first_indices + batch_steps) % n_a

    #generate shifts of first_input and second_input from indices using noise dist
    first_noise = noise_dist.sample(first_indices.shape) * n_a / (2* pi)
    second_noise = noise_dist.sample(second_indices.shape) * n_a / (2* pi)

    first_input = (first_indices + first_noise)
    second_input = (second_indices + second_noise)
    #first_input = (first_indices + sigma_1) % n_a
    #second_input = (second_indices + sigma_2) % n_a


    rnn.initialise_u(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)[:,:-1]  # Exclude fixation unit for prestimulus input
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(prestim_sensory_input, eta, False)

    stim_sensory_input = generate_sensory_input_spatial(first_input, True, n_a, A, kappa)[:,:-1]  # Exclude fixation unit for stimulus input
    for ts in range(stim_timesteps_1):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)[:,:-1]  # Exclude fixation unit for delay input
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_sensory_input, eta, False)

    stim_sensory_input = generate_sensory_input_spatial(second_input, True, n_a, A, kappa)[:,:-1]  # Exclude fixation unit for second stimulus input
    for ts in range(stim_timesteps_2):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)[:,:-1]  # Exclude fixation unit for delay input
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_sensory_input, eta, False)

    batch_network_outputs = []

    resp_sensory_input = generate_blank_sensory_input(n_a, False, batch_size)[:,:-1]  # Exclude fixation unit for response input
    for ts in range(resp_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
        batch_network_outputs.append(network_output)

    all_network_outputs = torch.stack(batch_network_outputs, 1)
    loss = loss_function_spatial(second_indices, all_network_outputs, n_a)
    #loss = loss_function_cosine_similarity(second_indices, all_network_outputs, n_a)
    x_err, y_err = errors_spatial(second_indices, all_network_outputs.detach(), n_a)
    
    # Compute ideal observer loss for comparison
    ideal_estimates = compute_ideal_observer_estimates(first_input, second_input, kappa_tilde, p)
    # Convert ideal estimates to 2D output format for loss computation
    ideal_angles = ideal_estimates * 2 * pi / n_a
    ideal_x = torch.cos(ideal_angles).unsqueeze(1).repeat(1, resp_timesteps).unsqueeze(2)
    ideal_y = torch.sin(ideal_angles).unsqueeze(1).repeat(1, resp_timesteps).unsqueeze(2)
    ideal_outputs = torch.cat([ideal_x, ideal_y], dim=2)
    ideal_loss = loss_function_spatial(second_indices, ideal_outputs, n_a)
    
    loss.backward()
    opt.step()

    losses.append(loss.item())
    network_losses.append(loss.item())
    optimal_losses.append(ideal_loss.item())
    dim1_errs.append(x_err.item())
    dim2_errs.append(y_err.item())

    if b % 200 == 0:
        plt.close('all')
        fig, axes = plt.subplots(4, figsize=(10, 12))

        axes[0].plot(losses[50:], label='Network Loss')
        if len(optimal_losses) > 50:
            axes[0].plot(optimal_losses[50:], label='Ideal Observer Loss')
        axes[0].legend()
        axes[0].set_title('Loss Comparison')
        
        axes[1].plot(dim1_errs[50:])
        axes[1].set_title('Angle Errors')
        
        axes[2].plot(dim2_errs[50:])
        axes[2].set_title('Magnitude Errors')
        
        # Plot loss gap
        if len(optimal_losses) > 50:
            loss_gap = np.array(network_losses[50:]) - np.array(optimal_losses[50:])
            axes[3].plot(loss_gap)
            axes[3].set_title('Performance Gap (Network - Ideal)')
            axes[3].set_ylabel('Loss Difference')

        print(f"Batch {b}: Network Loss: {losses[-1]:.4f}, Ideal Loss: {optimal_losses[-1]:.4f}")
        fig.savefig('section_chris/grid_losses.png')

# Final comparison plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(network_losses, label='Network Loss', alpha=0.7)
plt.plot(optimal_losses, label='Ideal Observer Loss', alpha=0.7)
plt.xlabel('Training Batch')
plt.ylabel('Loss')
plt.legend()
plt.title('Network vs Ideal Observer Performance')

plt.subplot(2, 2, 2)
loss_gap = np.array(network_losses) - np.array(optimal_losses)
plt.plot(loss_gap)
plt.xlabel('Training Batch')
plt.ylabel('Loss Difference')
plt.title('Performance Gap (Network - Ideal)')

plt.subplot(2, 2, 3)
plt.plot(network_losses, label='Network Loss', alpha=0.7)
plt.plot(optimal_losses, label='Ideal Observer Loss', alpha=0.7)
plt.xlabel('Training Batch')
plt.ylabel('Loss')
plt.legend()
plt.title('Network vs Ideal Observer Performance (Log Scale)')
plt.yscale('log')

plt.subplot(2, 2, 4)
# Rolling average of gap
window_size = 100
if len(loss_gap) > window_size:
    rolling_gap = np.convolve(loss_gap, np.ones(window_size)/window_size, mode='valid')
    plt.plot(rolling_gap)
    plt.xlabel('Training Batch')
    plt.ylabel('Rolling Average Loss Difference')
    plt.title(f'Smoothed Performance Gap (window={window_size})')

plt.tight_layout()
plt.savefig('section_chris/training_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Final Network Loss: {network_losses[-1]:.6f}")
print(f"Final Ideal Observer Loss: {optimal_losses[-1]:.6f}")
print(f"Final Performance Gap: {loss_gap[-1]:.6f}")
print(f"Mean Performance Gap: {np.mean(loss_gap):.6f}")
print(f"Std Performance Gap: {np.std(loss_gap):.6f}")


# In[58]:


### RUN TEST TRIAL FOR ACTIVITY
#target_indices =  torch.tensor([[i,j] for i in range(1,n_a+1) for j in range(1,n_a+1)])
first_indices = torch.rand(1000,) * n_a
second_indices = torch.rand(1000,) * n_a
test_batch_size = first_indices.shape[0]
batch_steps = torch.randint(6, first_indices.size())
#second_indices = (first_indices + steps[batch_steps]) % n_a
rnn.initialise_u(test_batch_size)
eta_tilde = randn(test_batch_size, N)
#same_different = generate_same_different(test_batch_size)
#second_stim = distractor_indices #torch.where(same_different==1,target_indices, distractor_indices)

#sigma_1 = torch.randn_like(target_indices) * 0.2
#sigma_2 = torch.randn_like(distractor_indices) * 0.2

#target_indices #+= sigma_1
#second_stim# += sigma_2

prestim_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)[:,:-1]  # Exclude fixation unit for prestimulus input
for ts in range(prestim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(prestim_sensory_input, eta, False)

stim_sensory_input = generate_sensory_input_spatial(first_indices, True, n_a, A, kappa)[:,:-1]
for ts in range(stim_timesteps_1):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(stim_sensory_input, eta, False)

delay_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)[:,:-1]  # Exclude fixation unit for delay input
for ts in range(delay_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(delay_sensory_input, eta, False)

stim_sensory_input = generate_sensory_input_spatial(second_indices, True, n_a, A, kappa)[:,:-1]  # Exclude fixation unit for second stimulus input
for ts in range(stim_timesteps_2):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(stim_sensory_input, eta, False)

delay_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)[:,:-1]  # Exclude fixation unit for delay input
for ts in range(delay_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(delay_sensory_input, eta, False)

test_trial_voltages = []
test_trial_outputs = []

resp_sensory_input = generate_blank_sensory_input(n_a, False, test_batch_size)[:,:-1]  # Exclude fixation unit for response input
for ts in range(resp_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    voltage, network_output = rnn.step_dynamics(resp_sensory_input, eta, return_output=True)
    test_trial_voltages.append(voltage)
    test_trial_outputs.append(network_output)

all_test_trial_voltages = torch.stack(test_trial_voltages, 1).detach()

# In[59]:


test_trial_outputs_tensor = torch.stack(test_trial_outputs, dim=1).detach()  # [batch, time, 2]

# Step 1: Normalize each 2D output vector
unit_outputs = test_trial_outputs_tensor #test_trial_outputs_tensor / test_trial_outputs_tensor.norm(dim=2, keepdim=True)

# Step 2: Compute mean vector over time
mean_vector = unit_outputs.mean(dim=1)  # [batch, 2]

# Step 3: Convert mean vector to angle
x, y = mean_vector[:, 0], mean_vector[:, 1]
theta_hat = torch.atan2(y, x) % (2 * torch.pi)  # ensure theta in [0, 2π)


theta = second_indices * 2 * torch.pi / n_a
theta_2 = first_indices * 2 * torch.pi / n_a

#theta_2 = torch.rand(1000,) * 2 * torch.pi


# %%
fig, ax = plt.subplots(2)
ax[0].scatter(theta,theta_hat,c=steps[batch_steps])
ax[1].scatter(theta_2,theta_hat,c=steps[batch_steps])


 # %%
def circular_distance(b1, b2):
    r = (b2 - b1) % (2*torch.pi)
    r = torch.where(r >= torch.pi, r - 2*torch.pi, r)
    return r

# %%
# Compute signed angle differences
delta_theta = circular_distance(theta_2, theta)        # Direction θ₂ - θ₁
error_theta = circular_distance(theta_hat, theta)      # Direction θ̂ - θ₁

error_theta = torch.where(delta_theta < 0, -1*error_theta, error_theta)
delta_theta = torch.where(delta_theta < 0, -1*delta_theta, delta_theta)

# %%
#same_mask = same_different == 1
#diff_mask = same_different == 2

plt.figure(figsize=(6, 4))
plt.scatter(delta_theta.numpy(), error_theta.numpy(), c=steps[batch_steps], alpha=0.7)
#plt.scatter(delta_theta[diff_mask].numpy(), error_theta[diff_mask].numpy(), color='tab:orange', label='Different cause', alpha=0.7)

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('distance θ₂ - θ₁ (rad)')
plt.ylabel('Signed error θ (rad)')
plt.title('Response Bias Relative to Distractor Stimulus')
plt.grid(True)
plt.tight_layout()
plt.show()

 # %%
df = pd.DataFrame({"delta_theta": delta_theta, "error_theta": error_theta})
sns.regplot(x="delta_theta",y="error_theta",data=df,x_bins=np.arange(0, 2*torch.pi, torch.pi/10), scatter=True, fit_reg=False)


# %%

# Inverse logit (logistic function)
def inv_logit(x):
    return 1 / (1 + np.exp(-x))

# Estimate shared likelihood
def estimate_shared_likelihood(x1, x2, kappa):
    delta_x = np.abs(x1 - x2)
    delta_x = np.where(delta_x > np.pi, 2*np.pi - delta_x, delta_x)
    kappa_eff = 2 * kappa * np.cos(delta_x / 2)
    marginal_likelihood = besselI(0, kappa_eff) / ((2 * np.pi)**2 * besselI(0, kappa)**2)
    return {"lik_shared": marginal_likelihood, "kappa_eff": kappa_eff}

# Compute posterior probability of shared cause
def compute_p_shared(x1, x2, kappa, p=0.5):
    lik_estimate = estimate_shared_likelihood(x1, x2, kappa)
    lik_shared = lik_estimate["lik_shared"]
    lik_indep = (1 / (2 * np.pi))**2

    log_p1 = np.log(p) + np.log(lik_shared)
    log_p2 = np.log(1 - p) + np.log(lik_indep)

    posterior_prob_shared = inv_logit(log_p1 - log_p2)
    return posterior_prob_shared

# MAP estimate from mixture of von Mises
def calc_map(theta, kappa_1, kappa_2, mu_1, mu_2):
    def d_mixed(x):
        return theta * vonmises.pdf(x, kappa_1, loc=mu_1) + (1 - theta) * vonmises.pdf(x, kappa_2, loc=mu_2)

    def neg_d_mixed(x):
        return -d_mixed(x)

    result = minimize_scalar(neg_d_mixed, bounds=(-np.pi, np.pi), method='bounded')
    return result.x

# %%
# Compute MAP estimate for each pair of angles
map_estimates = []
for i in range(len(theta)):
    p_shared = compute_p_shared(theta_2[i], theta[i], kappa_tilde, p)
    mu_shared = (theta_2[i] + theta[i]) / 2
    mu_shared = mu_shared % (2 * np.pi)  # Ensure mu_shared is within [0, 2π]
    kappa_eff = 2 * kappa_tilde * np.cos((theta_2[i] - theta[i]) / 2)
    map_estimate = calc_map(p_shared, kappa_eff, kappa_tilde, mu_shared, theta[i])
    map_estimate = map_estimate % (2 * np.pi)  # Ensure map_estimate is within [0, 2π]
    map_estimates.append(map_estimate)



# %%
# plot the MAP estimates against the theta_hat
plt.figure(figsize=(6, 4))
plt.scatter(theta_hat.numpy(), map_estimates, c=steps[batch_steps], alpha=0.7)
plt.xlabel('Response θ̂ (rad)')
plt.ylabel('MAP Estimate θ (rad)')
plt.title('MAP Estimates vs Response Angles')


# %%
from scipy.stats import vonmises
from scipy.integrate import quad
import numpy as np
circular_means = []
def calc_circular_mean(theta, kappa_1, kappa_2, mu_1, mu_2):
    # Mixture PDF
    def d_mixed(x):
        return theta * vonmises.pdf(x, kappa_1, loc=mu_1) + (1 - theta) * vonmises.pdf(x, kappa_2, loc=mu_2)

    # Integrate sin(x) * f(x) and cos(x) * f(x)
    sin_integral = quad(lambda x: np.sin(x) * d_mixed(x), -np.pi, np.pi, limit=100)[0]
    cos_integral = quad(lambda x: np.cos(x) * d_mixed(x), -np.pi, np.pi, limit=100)[0]

    # Compute circular mean
    circular_mean = np.arctan2(sin_integral, cos_integral)
    return circular_mean

def calc_circular_mean_stable(theta, kappa_1, kappa_2, mu_1, mu_2):
    # Use scipy's vonmises for better numerical stability
    from scipy.stats import vonmises
    
    # For circular mean of mixture, use the formula:
    # E[e^(ix)] = theta * E_1[e^(ix)] + (1-theta) * E_2[e^(ix)]
    # where E_j[e^(ix)] = I_1(kappa_j)/I_0(kappa_j) * e^(i*mu_j)
    
    # Compute characteristic functions
    r1 = besselI(1, kappa_1) / besselI(0, kappa_1)  # Mean resultant length
    r2 = besselI(1, kappa_2) / besselI(0, kappa_2)
    
    # Complex representation
    z1 = r1 * np.exp(1j * mu_1)
    z2 = r2 * np.exp(1j * mu_2)
    
    # Mixture characteristic function
    z_mix = theta * z1 + (1 - theta) * z2
    
    # Extract circular mean
    return np.angle(z_mix)

# Compute circular mean for each pair of angles
post_circ_means = []
for i in range(len(theta)):
    p_shared = compute_p_shared(theta_2[i], theta[i], kappa_tilde, p)
    mu_shared = compute_circular_mean(theta_2[i], theta[i])
    mu_shared = mu_shared % (2 * np.pi)
    delta_x = abs(theta_2[i] - theta[i])
    delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
    kappa_eff = 2 * kappa_tilde * np.cos(delta_x / 2)
    circ_mean = calc_circular_mean_stable(p_shared, kappa_eff, kappa_tilde, mu_shared, theta[i])
    circ_mean = circ_mean % (2 * np.pi)  # Ensure circ_mean is within [0, 2π]
    post_circ_means.append(circ_mean)


# %% curve of MAP estimate and posterior circular mean from 0 to pi
map_estimate_curve = []
post_circ_mean_curve = []
delta_x = np.linspace(0, np.pi, 1000)

for x in delta_x:
    mu_shared = (0 + x) / 2
    p_shared = compute_p_shared(0, x, kappa_tilde, p)
    kappa_eff = 2 * kappa_tilde * np.cos((0 - x) / 2)
    map_estimate_curve.append(calc_map(p_shared, kappa_eff, kappa_tilde, mu_shared, 0))
    post_circ_mean_curve.append(calc_circular_mean_stable(p_shared, kappa_eff, kappa_tilde, mu_shared, 0))

# %%
plt.plot(delta_x, map_estimate_curve, color='red', linestyle='--', label='MAP')
plt.plot(delta_x, post_circ_mean_curve, color='blue', linestyle='--', label='Posterior Circular Mean')
plt.scatter(delta_theta.numpy(), error_theta.numpy(), alpha=0.7, label='Data Points')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlim(0, np.pi)
plt.xlabel('Delta θ̂(rad)')
plt.ylabel('Bias θ (rad)')
plt.title('MAP and Circular Mean Curves vs Response Angles')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
# plot the circular mean and the binned error_theta (as in the sns regplot above)
plt.figure(figsize=(6, 4))
sns.regplot(x="delta_theta", y="error_theta", data=df, x_bins=np.arange(0, 2*torch.pi, torch.pi/10), scatter=True, fit_reg=False)
plt.plot(delta_x, post_circ_mean_curve, color='blue', linestyle='--', label='Posterior Circular Mean')
plt.xlabel('Distance θ₂ - θ₁ (rad)')
plt.ylabel('Signed Error θ (rad)')
plt.title('y_hat and posterior central tendencies')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
def circular_mean(angles: torch.Tensor) -> torch.Tensor:
    sin_sum = torch.sin(angles).mean(dim=0)
    cos_sum = torch.cos(angles).mean(dim=0)
    return torch.atan2(sin_sum, cos_sum)


sorted_idx = np.argsort(delta_theta)
delta_theta_sorted = delta_theta[sorted_idx]
error_theta_sorted = error_theta[sorted_idx]

# Parameters
window_width = np.pi / 5   # e.g. window size ~18°
step_size = np.pi / 400     # e.g. step ~1.8°
x_vals = np.arange(0, np.pi, step_size)

# Compute sliding circular mean
circ_mean_vals = []

for x in x_vals:
    # Create sliding window
    lower = x - window_width / 2
    upper = x + window_width / 2

    # Handle circular wraparound (use modular arithmetic)
    if lower < 0:
        mask = (delta_theta_sorted >= (lower + 2 * np.pi)) | (delta_theta_sorted < upper)
    elif upper > 2 * np.pi:
        mask = (delta_theta_sorted >= lower) | (delta_theta_sorted < (upper - 2 * np.pi))
    else:
        mask = (delta_theta_sorted >= lower) & (delta_theta_sorted < upper)

    values = error_theta_sorted[mask]
    if len(values) == 0:
        circ_mean_vals.append(np.nan)
    else:
        circ_mean = circular_mean(values)
        circ_mean_vals.append(circ_mean)

circ_mean_vals = np.array(circ_mean_vals)
# %%
plt.figure(figsize=(6, 4))

# Scatter points
#plt.scatter(delta_theta, error_theta, alpha=0.3, label='Raw data')

# Circular mean per bin
plt.plot(x_vals, circ_mean_vals, color='blue', linestyle='--', label='y_hat (sliding window)')
plt.plot(delta_x, post_circ_mean_curve, color='red', linestyle='--', label='Posterior Circular Mean')
plt.plot(delta_x, map_estimate_curve, color='green', linestyle='--', label='MAP Estimate')
plt.xlabel('Distance θ₂ - θ₁ (rad)')
plt.ylabel('Signed Error θ (rad)')
plt.title('y_hat and posterior central tendencies')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%

def func_scalar(x, kappa, p):
        p_shared = compute_p_shared(0, x, kappa, p)
        kappa_eff = 2 * kappa * np.cos((0 - x) / 2)
        result = calc_circular_mean_stable(p_shared, kappa_eff, kappa, x / 2, 0)
        return result

func = np.vectorize(func_scalar)

xdata = delta_theta.numpy()
ydata = error_theta.numpy()
# Fit the function to the data
from scipy.optimize import curve_fit

popt, pcov = curve_fit(func, delta_theta.numpy(), error_theta.numpy(), p0=[kappa_tilde, p], bounds=([0, 0], [np.inf, 1]),loss="soft_l1")


# %%
plt.plot(delta_theta.numpy(), error_theta.numpy(), 'o', label='Data', alpha=0.7)
plt.plot(delta_theta.numpy()[np.argsort(delta_theta.numpy())], func(delta_theta.numpy()[np.argsort(delta_theta.numpy())], *popt), 'r-', label='Fitted function, kappa={:.2f}, p={:.2f}'.format(popt[0], popt[1]))
plt.plot(delta_theta.numpy()[np.argsort(delta_theta.numpy())], func(delta_theta.numpy()[np.argsort(delta_theta.numpy())], kappa_tilde, p), 'b--', label='Posterior Circular Mean (kappa={:.2f}, p={:.2f})'.format(kappa_tilde, p))
plt.xlabel('Distance θ₂ - θ₁ (rad)')
plt.ylabel('Signed Error θ (rad)')
plt.title('Fitted Function to Data: recurrent network')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# After your curve_fit call (around line 678)
popt, pcov = curve_fit(func, delta_theta.numpy(), error_theta.numpy(), p0=[kappa_tilde, p], bounds=([0, 0], [np.inf, 1]),loss="soft_l1")

# Compute predictions for both parameter sets
fitted_predictions = func(delta_theta.numpy(), *popt)
putative_predictions = func(delta_theta.numpy(), kappa_tilde, p)

# Calculate losses (sum of squared residuals)
fitted_loss = np.sum((error_theta.numpy() - fitted_predictions)**2)
putative_loss = np.sum((error_theta.numpy() - putative_predictions)**2)

# Calculate R-squared for both
def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

fitted_r2 = calculate_r_squared(error_theta.numpy(), fitted_predictions)
putative_r2 = calculate_r_squared(error_theta.numpy(), putative_predictions)

# Print comparison
print(f"Fitted parameters: κ={popt[0]:.3f}, p={popt[1]:.3f}")
print(f"Putative parameters: κ={kappa_tilde:.3f}, p={p:.3f}")
print(f"")
print(f"Fitted loss (SSE): {fitted_loss:.6f}")
print(f"Putative loss (SSE): {putative_loss:.6f}")
print(f"Loss improvement: {((putative_loss - fitted_loss) / putative_loss * 100):.2f}%")
print(f"")
print(f"Fitted R²: {fitted_r2:.4f}")
print(f"Putative R²: {putative_r2:.4f}")
# %%
