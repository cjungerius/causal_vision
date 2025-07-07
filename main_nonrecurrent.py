# %%
import torch
from tqdm import tqdm
from section_6.utils import generate_sensory_input_spatial
from section_chris.utils_chris import generate_batch_of_continuous_wm_targets, loss_function_spatial_non_recurrent, generate_same_different
import numpy as np
from scipy.special import iv as besselI
from torch.distributions import VonMises
import matplotlib.pyplot as plt

# %%
## Define our task
n_a = 2   # points along circular dimension
A = 1.0
kappa = 3

kappa_tilde = 3 # sensory noise
p = 4/5  # probability of same vs different
noise_dist = VonMises(0, kappa_tilde)

# %%
# Define our network: a simple fully connected network (not recurrent this time!)
class MyNetwork(torch.nn.Module):
    def __init__(self, n_a):
        super(MyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(n_a*2, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# %% Initialize the network
model = MyNetwork(n_a)

# %%
# Training parameters
batch_size = 200
learning_rate = 0.001
num_batches = 2000
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %% set up for tracking loss and ideal observer loss
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
        estimates.append(circ_mean)
    
    return torch.tensor(estimates)

network_losses = []  # Store for network loss
optimal_losses = []  # Store for optimal loss

# %%# Training loop
for b in tqdm(range(num_batches)):
    model.train()
    optimizer.zero_grad()
    
    # Generate input data: a von mises bump
    x_1 = generate_batch_of_continuous_wm_targets(n_a, batch_size)
    x_2 = generate_batch_of_continuous_wm_targets(n_a, batch_size)
    same_different = generate_same_different(batch_size, p)
    x_2 = torch.where(same_different==1,x_1, x_2)

    # add sensory noise
    x_1_tilde = x_1 + (noise_dist.sample(x_1.shape) / (2 * torch.pi) * n_a)  # Scale noise to match the range of x_2
    x_2_tilde = x_2 + (noise_dist.sample(x_2.shape) / (2 * torch.pi) * n_a)  # Scale noise to match the range of x_2

    x_1_spatial = torch.stack([torch.cos(x_1_tilde * 2 * np.pi / n_a), torch.sin(x_1_tilde * 2 * np.pi / n_a)], dim=1)
    x_2_spatial = torch.stack([torch.cos(x_2_tilde * 2 * np.pi / n_a), torch.sin(x_2_tilde * 2 * np.pi / n_a)], dim=1)
    #x_1_spatial = generate_sensory_input_spatial(x_1_tilde, False, n_a, A, kappa)[:,:-1]
    #x_2_spatial = generate_sensory_input_spatial(x_2_tilde, False, n_a, A, kappa)[:,:-1]
    x = torch.cat((x_1_spatial, x_2_spatial), dim=1)  # Concatenate the two inputs
    
    # Forward pass
    output = model(x)
    loss = loss_function_spatial_non_recurrent(x_2, output, n_a)
    
    # Compute ideal observer loss
    ideal_angles = compute_ideal_observer_estimates(x_1_tilde, x_2_tilde, kappa_tilde, p)
    ideal_x = torch.cos(ideal_angles).unsqueeze(-1)
    ideal_y = torch.sin(ideal_angles).unsqueeze(-1)
    ideal_outputs = torch.cat([ideal_x, ideal_y], dim=1)
    ideal_observer_loss = loss_function_spatial_non_recurrent(x_2, ideal_outputs, n_a)

    # Store losses for analysis
    network_losses.append(loss.item())
    optimal_losses.append(ideal_observer_loss.item())

    if b % 200 == 0:
        plt.close('all')
        fig, axes = plt.subplots(2, figsize=(10, 12))

        axes[0].plot(network_losses, label='Network Loss')
        axes[0].plot(optimal_losses, label='Ideal Observer Loss')
        axes[0].legend()
        axes[0].set_title('Loss Comparison')
        
        # Plot loss gap
        loss_gap = np.array(network_losses) - np.array(optimal_losses)
        axes[1].plot(loss_gap)
        axes[1].set_title('Performance Gap (Network - Ideal)')
        axes[1].set_ylabel('Loss Difference')

        print(f"Batch {b}: Network Loss: {network_losses[-1]:.4f}, Ideal Loss: {optimal_losses[-1]:.4f}")
        fig.savefig('section_chris/non_recurrent_losses.png')

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

# %% run test trials to visualize the network output
model.eval()
test_batch_size = 1000

x_1 = generate_batch_of_continuous_wm_targets(n_a, test_batch_size)
x_2 = generate_batch_of_continuous_wm_targets(n_a, test_batch_size)

x_1_spatial = torch.stack([torch.cos(x_1 * 2 * np.pi / n_a), torch.sin(x_1 * 2 * np.pi / n_a)], dim=1)
x_2_spatial = torch.stack([torch.cos(x_2 * 2 * np.pi / n_a), torch.sin(x_2 * 2 * np.pi / n_a)], dim=1)
#x_1_spatial = generate_sensory_input_spatial(x_1, False, n_a, A, kappa)[:,:-1]
#x_2_spatial = generate_sensory_input_spatial(x_2, False, n_a, A, kappa)[:,:-1]
x = torch.cat((x_1_spatial, x_2_spatial), dim=1)  # Concatenate the two inputs

output = model(x)



# %%
theta_target = x_2 * 2 * torch.pi / n_a
theta_distractor = x_1 * 2 * torch.pi / n_a
theta_output = torch.atan2(output[:,1], output[:,0]).detach() % (2 * torch.pi)

# %%
def circular_distance(b1, b2):
    r = (b2 - b1) % (2*torch.pi)
    r = torch.where(r >= torch.pi, r - 2*torch.pi, r)
    return r
# %%
# Compute signed angle differences
delta_theta = circular_distance(theta_distractor, theta_target)        # Direction θ₂ - θ₁
error_theta = circular_distance(theta_output, theta_target)      # Direction θ̂ - θ₁

error_theta = torch.where(delta_theta < 0, -1*error_theta, error_theta)
delta_theta = torch.where(delta_theta < 0, -1*delta_theta, delta_theta)


# %%
plt.figure(figsize=(6, 4))
plt.scatter(delta_theta.numpy(), error_theta.numpy(), alpha=0.7)

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('distance θ₂ - θ₁ (rad)')
plt.ylabel('Signed error θ (rad)')
plt.title('Response Bias Relative to Distractor Stimulus')
plt.grid(True)
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

popt, pcov = curve_fit(func, delta_theta.numpy(), error_theta.numpy(), p0=[kappa_tilde, p], bounds=([0, 0], [np.inf, 1]))


# %%
plt.plot(delta_theta.numpy(), error_theta.numpy(), 'o', label='Data',alpha=0.7)
plt.plot(delta_theta.numpy()[np.argsort(delta_theta.numpy())], func(delta_theta.numpy()[np.argsort(delta_theta.numpy())], *popt), 'r-', label='Fitted function, kappa={:.2f}, p={:.2f}'.format(popt[0], popt[1]))
plt.plot(delta_theta.numpy()[np.argsort(delta_theta.numpy())], func(delta_theta.numpy()[np.argsort(delta_theta.numpy())], kappa_tilde, p), 'b--', label='Posterior Circular Mean (kappa={:.2f}, p={:.2f})'.format(kappa_tilde, p))
plt.xlabel('Distance θ₂ - θ₁ (rad)')
plt.ylabel('Signed Error θ (rad)')
plt.title('Fitted Function to Data: simple non-recurrent network')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
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
