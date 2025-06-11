# coding: utf-8

# In[1]:

# %matplotlib widget
from section_6.rate_rnn_with_io import ReluRateRNNWithIO
from section_6.utils import generate_blank_sensory_input, generate_sensory_input_spatial, loss_function_spatial, errors_spatial, loss_function_non_spatial
from section_chris.utils_chris import generate_batch_of_continuous_wm_targets, generate_sensory_input_2d_vonmises, loss_function_2d_vonmises, errors_spatial_2d, generate_same_different
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
n_a = 10    # points along circular dimension
A = 1.0
kappa = 5.0


# In[53]:


## Define our system
dt = 0.01
tau = 0.2
N = 40
batch_size = 48
rnn = ReluRateRNNWithIO(dt, N, n_a + 1, 2, tau)    # Refer to notes to understand why the i/o are these sizes!


# In[54]:


## Define training machinery
lr = 1e-3
opt = Adam(rnn.parameters(), lr)
num_batches = 8000
losses = [] # Store for loss curve plotting
dim1_errs = [] # Store for accuracies curve plotting
dim2_errs = []


# In[55]:


## Define simulation parameters
T_prestim = 0.1     # in seconds
T_stim_1 = 0.4
T_stim_2 = 0.4
T_delay = 0.0
T_resp = 0.1

prestim_timesteps = int(T_prestim / dt)
stim_timesteps_1 = int(T_stim_1 / dt)
stim_timesteps_2 = int(T_stim_2 / dt)
delay_timesteps = int(T_delay / dt)
resp_timesteps = int(T_resp / dt)

kappa_tilde = 10 # von Mises concentration parameter for the sensory input
# noise generating von mises distribution
noise_dist = VonMises(0, kappa_tilde)
p = 0.7

eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = .1

steps = torch.tensor([20, 35, 75, -20, -35, -75]) / 360 * n_a

## Initialise simulation, including the first noise term
eta_tilde = randn(batch_size, N)


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

    first_input = (first_indices + first_noise) % n_a
    second_input = (second_indices + second_noise) % n_a
    #first_input = (first_indices + sigma_1) % n_a
    #second_input = (second_indices + sigma_2) % n_a


    rnn.initialise_u(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(prestim_sensory_input, eta, False)

    stim_sensory_input = generate_sensory_input_spatial(first_input, True, n_a, A, kappa)
    for ts in range(stim_timesteps_1):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N))
        eta = eta_tilde * C
        rnn.step_dynamics(delay_sensory_input, eta, False)

    stim_sensory_input = generate_sensory_input_spatial(second_input, True, n_a, A, kappa)
    for ts in range(stim_timesteps_2):
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
    loss = loss_function_spatial(second_indices, all_network_outputs, n_a)
    x_err, y_err = errors_spatial(second_indices, all_network_outputs.detach(), n_a)
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

prestim_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)
for ts in range(prestim_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(prestim_sensory_input, eta, False)

stim_sensory_input = generate_sensory_input_spatial(first_indices, True, n_a, A, kappa)
for ts in range(stim_timesteps_1):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(stim_sensory_input, eta, False)

delay_sensory_input = generate_blank_sensory_input(n_a, True, test_batch_size)
for ts in range(delay_timesteps):
    eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(test_batch_size, N))
    eta = eta_tilde * C
    rnn.step_dynamics(delay_sensory_input, eta, False)

stim_sensory_input = generate_sensory_input_spatial(second_indices, True, n_a, A, kappa)
for ts in range(stim_timesteps_2):
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
x, y = y_hat[:, 0], y_hat[:, 1]
theta_hat = torch.atan2(y, x) % (2 * torch.pi)

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


# %% curve of MAP estimate and posterior circular mean from 0 to 2*pi
map_estimate_curve = []
post_circ_mean_curve = []
delta_x = np.linspace(0, np.pi, 1000)

for x in delta_x:
    mu_shared = (0 + x) / 2
    p_shared = compute_p_shared(0, x, kappa_tilde, p)
    kappa_eff = 2 * kappa_tilde * np.cos((0 - x) / 2)
    map_estimate_curve.append(calc_map(p_shared, kappa_eff, kappa_tilde, mu_shared, 0))
    post_circ_mean_curve.append(calc_circular_mean(p_shared, kappa_eff, kappa_tilde, mu_shared, 0))

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
sns.regplot(x="delta_theta", y="error_theta", data=df, x_bins=np.arange(0, 2*torch.pi, torch.pi/20), scatter=True, fit_reg=False)
plt.plot(delta_x, post_circ_mean_curve, color='blue', linestyle='--', label='Posterior Circular Mean')
plt.xlabel('Distance θ₂ - θ₁ (rad)')
plt.ylabel('Signed Error θ (rad)')
plt.title('Posterior Circular Mean vs Binned Error')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%
