# %%
import numpy as np
from scipy.special import iv as besselI
import torch
from torch import randn
from torch.distributions import VonMises
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import matplotlib.pyplot as plt
from section_chris.ReluRNNLayer import MyRNN
from section_6.utils import generate_blank_sensory_input
from skimage.color import lab2rgb
import pandas as pd
import seaborn as sns
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Load the ConvNeXt model with pretrained weights
model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
# Set the model to evaluation mode
model.eval()
weights = ConvNeXt_Base_Weights.DEFAULT
preprocess = weights.transforms()
model.features = model.features[0:2]  # Use only the first two layers for feature extraction
model.to(device)

# %%

def gabor(size, sigma, theta, Lambda, psi, gamma):
    """Draw a gabor patch."""
    sigma_x = sigma
    sigma_y = sigma / gamma
    Lambda = torch.tensor(Lambda)
    psi = torch.tensor(psi)
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta)

    # Bounding box
    s = torch.linspace(-1, 1, size)

    (x, y) = torch.meshgrid(s, s, indexing='ij')

    # Rotation
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    gb = torch.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * torch.cos(2 * torch.pi / Lambda * x_theta + psi)
    return gb

# %%
def generate_batch(batch_size, p=0.5, kappa=5.0, q=0.0):
    """Generate a batch of Gabor patches."""

    if p == -1:
        # testing mode: no noise
        target_angles = torch.rand(batch_size) * 2 * torch.pi
        colors_1 = torch.rand(batch_size) * 2 * torch.pi
        colors_2 = torch.rand(batch_size) * 2 * torch.pi
        distractor_angles = torch.rand(batch_size) * 2 * torch.pi
        angles_1 = target_angles.clone().detach()
        angles_2 = distractor_angles.clone().detach()

    else:
        # Von Mises noise distribution
        vm = VonMises(0, kappa)

        # Generate ground truth angles
        target_angles = torch.rand(batch_size) * 2 * torch.pi
        target_colors = torch.rand(batch_size) * 2 * torch.pi
        # Generate distractor angles
        distractor_angles = torch.rand(batch_size) * 2 * torch.pi
        distractor_colors = torch.rand(batch_size) * 2 * torch.pi
        
        p_color_match = p + q * (1-p)
        p_color_diff = p - q * (1-p)

        match_ori = torch.rand(batch_size) < p
        distractor_angles = torch.where(match_ori, target_angles, distractor_angles)

        # Decide whether the second pair matches
        rand_color = torch.rand(batch_size)
        match_color = torch.where(
            match_ori,
            rand_color < p_color_match,
            rand_color < p_color_diff
        )

        distractor_colors = torch.where(match_color, target_colors, distractor_colors)

        # Add noise to the target angles
        noise = vm.sample((batch_size,))
        angles_1 = (target_angles + noise) % (2 * torch.pi)  
        noise = vm.sample((batch_size,))
        colors_1 = (target_colors + noise) % (2 * torch.pi)
        # Add noise to the distractor angles
        noise = vm.sample((batch_size,))
        angles_2 = (distractor_angles + noise) % (2 * torch.pi) 
        noise = vm.sample((batch_size,))
        colors_2 = (distractor_colors + noise) % (2 * torch.pi)

    # adjust for testing mode: distractor colors are always the same as target colors
    if q == 1:
        colors_2 = colors_1.clone().detach()
        print("colors cloned!")
    elif q == -1:
        colors_2 = (colors_1 + torch.pi) % (2 * torch.pi)
        print("colors flipped!")

    # Convert to (C, H, W) format
    # Create Gabor patches for target and distractor angles
    target_gabors = torch.stack([gabor(232, sigma=.4, theta=angle/2, Lambda=.25, psi=0, gamma=1) for angle in angles_1])
    target_gabors += 1
    target_gabors /= 2
    target_gabors *= 74
    distractor_gabors = torch.stack([gabor(232, sigma=.4, theta=angle/2, Lambda=.25, psi=0, gamma=1) for angle in angles_2])
    distractor_gabors += 1
    distractor_gabors /= 2
    distractor_gabors *= 74
    # Convert to (C, H, W) format
    target_gabors = target_gabors.unsqueeze(1)  # Add channel dimension
    target_gabors = target_gabors.repeat(1, 3, 1, 1)  # Increase color dimensions to 3
    target_gabors[:,1,:,:] = torch.cos(colors_1).unsqueeze(-1).unsqueeze(-1) * 37
    target_gabors[:,2,:,:] = torch.sin(colors_1).unsqueeze(-1).unsqueeze(-1) * 37
    with  warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        target_gabors = lab2rgb(target_gabors.numpy(), channel_axis = 1)
    target_gabors = preprocess(torch.from_numpy(target_gabors))
    distractor_gabors = distractor_gabors.unsqueeze(1)  
    distractor_gabors = distractor_gabors.repeat(1, 3, 1, 1)
    distractor_gabors[:,1,:,:] = torch.cos(colors_2).unsqueeze(-1).unsqueeze(-1) * 37
    distractor_gabors[:,2,:,:] = torch.sin(colors_2).unsqueeze(-1).unsqueeze(-1) * 37
    with  warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        distractor_gabors = lab2rgb(distractor_gabors.numpy(), channel_axis = 1)
    distractor_gabors = preprocess(torch.from_numpy(distractor_gabors))

    with torch.no_grad():
        target_gabors = target_gabors.to(device)
        distractor_gabors = distractor_gabors.to(device)
        target_features = model.features(target_gabors)
        distractor_features = model.features(distractor_gabors)
        target_features = model.avgpool(target_features)
        distractor_features = model.avgpool(distractor_features)
        target_features = torch.flatten(target_features, 1)
        distractor_features = torch.flatten(distractor_features, 1)

    return target_features, distractor_features, target_angles, angles_1, angles_2, colors_1, colors_2

# %%

## Define our system
dt = 0.01
tau = 0.25  # time constant for the RNN dynamics
n_a = 128 # input size
N = 400 
batch_size = 100
rnn = MyRNN(n_a, N, 4, dt, tau)
rnn.to(device)
rnn.init_hidden(batch_size)

# load saved parameters if available
try:
    rnn.load_state_dict(torch.load("cnn_rnn_gabor_2d_model.pth", map_location=device))
    print("Loaded saved RNN parameters.")
except FileNotFoundError:
    print("No saved RNN parameters found. Starting with random initialization.")

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


# %%
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

    first_np = first_inputs.cpu().numpy()
    second_np = second_inputs.cpu().numpy()
    
    for i in range(len(first_np)):
        p_shared = compute_p_shared(first_np[i], second_np[i], kappa_tilde, p_prior)
        mu_shared = compute_circular_mean(first_np[i], second_np[i])
        mu_shared = mu_shared % (2 * np.pi)
        
        delta_x = abs(first_np[i] - second_np[i])
        delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
        kappa_eff = 2 * kappa_tilde * np.cos(delta_x / 2)
        
        circ_mean = calc_circular_mean_stable(p_shared, kappa_eff, kappa_tilde, mu_shared, first_np[i])
        circ_mean = circ_mean % (2 * np.pi)
        estimates.append(circ_mean)
    
    return torch.tensor(estimates)


# %%
# generate test batch

model.eval()
batch_size = 2000

eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = .01

## Initialise simulation, including the first noise term
eta_tilde = torch.randn(batch_size*2, N)
eta_tilde = eta_tilde.to(device)


#flip first and second here to change whether the network should focus on the first or second period (the first argument==the target period)
second_input, first_input, target_angles, second_angles, first_angles, second_colors, first_colors = generate_batch(batch_size, -1, kappa_tilde, 0)
first_input = first_input.to(device)
second_input = second_input.to(device)
target_angles = target_angles.to(device)
#concatenate with a batch where q == 1, so that the distractor colors are the same as the target colors
second_input_2, first_input_2, target_angles_2, second_angles_2, first_angles_2, second_colors_2, first_colors_2 = generate_batch(batch_size, -1, kappa_tilde, 0)
first_input_2 = first_input_2.to(device)
second_input_2 = second_input_2.to(device)
target_angles_2 = target_angles_2.to(device)
# concatenate the two batches
first_input = torch.cat((first_input, first_input_2), dim=0)
second_input = torch.cat((second_input, second_input_2), dim=0)
target_angles = torch.cat((target_angles, target_angles_2), dim=0)
second_angles = torch.cat((second_angles, second_angles_2), dim=0)
first_angles = torch.cat((first_angles, first_angles_2), dim=0)
second_colors = torch.cat((second_colors, second_colors_2), dim=0)
first_colors = torch.cat((first_colors, first_colors_2), dim=0)


batch_size = batch_size * 2  # since we concatenated two batches
# %%


with torch.no_grad():
    rnn.init_hidden(batch_size)

    prestim_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)[:,:-1].to(device)
    for ts in range(prestim_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N, device=device))
        eta = eta_tilde * C
        rnn.step(prestim_sensory_input, eta, False)

    stim_sensory_input = first_input
    for ts in range(stim_timesteps_1):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N, device=device))
        eta = eta_tilde * C
        rnn.step(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)[:,:-1].to(device)
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N, device=device))
        eta = eta_tilde * C
        rnn.step(delay_sensory_input, eta, False)

    stim_sensory_input = second_input
    for ts in range(stim_timesteps_2):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N, device=device))
        eta = eta_tilde * C
        rnn.step(stim_sensory_input, eta, False)

    delay_sensory_input = generate_blank_sensory_input(n_a, True, batch_size)[:,:-1].to(device)
    for ts in range(delay_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N, device=device))
        eta = eta_tilde * C
        rnn.step(delay_sensory_input, eta, False)

    batch_network_outputs = []

    resp_sensory_input = generate_blank_sensory_input(n_a, False, batch_size)[:,:-1].to(device)
    for ts in range(resp_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N, device=device))
        eta = eta_tilde * C
        voltage, network_output = rnn.step(resp_sensory_input, eta, return_output=True)
        batch_network_outputs.append(network_output)

    test_trial_outputs = torch.stack(batch_network_outputs, 1).detach().cpu()

# %%

mean_vector = test_trial_outputs.mean(dim=1)  # [batch, 2]

x, y = mean_vector[:, 0], mean_vector[:, 1]
theta_hat = torch.atan2(y, x) % (2 * torch.pi)  # ensure theta in [0, 2π)


theta = second_angles 
theta_2 = first_angles 

color = second_colors
color_2 = first_colors


# fig, ax = plt.subplots(2)
# ax[0].scatter(theta,theta_hat)
# ax[1].scatter(theta_2,theta_hat)

# %%
def circular_distance(b1, b2):
    r = (b2 - b1) % (2*torch.pi)
    r = torch.where(r >= torch.pi, r - 2*torch.pi, r)
    return r

# Compute signed angle differences
delta_theta = circular_distance(theta_2, theta)        # Direction θ₂ - θ₁
error_theta = circular_distance(theta_hat, theta)      # Direction θ̂ - θ₁
delta_color = circular_distance(color_2, color)            # Direction color₂ - color₁

error_theta = torch.where(delta_theta < 0, -1*error_theta, error_theta)
delta_theta = torch.abs(delta_theta)
delta_color = torch.abs(delta_color)
plt.scatter(delta_theta, delta_color, c=error_theta)
plt.colorbar()
plt.savefig("heatmap_thing.png")

x = delta_color
y = delta_theta
z = error_theta

vals, xbins, ybins = np.histogram2d(delta_color, delta_theta, weights = error_theta, bins=(30, 30))

m1 = plt.pcolormesh(ybins, xbins, vals, cmap='viridis')
plt.colorbar()
plt.savefig("another_heatmap.png")

# delta_color = torch.where(delta_color < 0, -1*delta_color, delta_color)
# color_near_far = torch.where(delta_color < torch.pi/2, 1, 0)  # 1 if near, 0 if far

# plt.figure(figsize=(6, 4))
# plt.scatter(delta_theta.numpy(), error_theta.numpy(), alpha=0.7, c=color_near_far)
# plt.xlim(-torch.pi, torch.pi)
# plt.ylim(-0.3, 0.3)

# plt.axhline(0, color='gray', linestyle='--')
# plt.xlabel('distance θ₂ - θ₁ (rad)')
# plt.ylabel('Signed error θ (rad)')
# plt.title('Response Bias Relative to Distractor Stimulus')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("model_analysis_figs/response_bias_relative_to_distractor_stimulus.png")
# # %%

# df = pd.DataFrame({"delta_theta": delta_theta, "error_theta": error_theta, "dist": color_near_far})
# # regplot split by dist
# sns.regplot(x="delta_theta", y="error_theta", data=df[df["dist"] == 1], x_bins=np.arange(-torch.pi, torch.pi, torch.pi/10), scatter=True, fit_reg=False, label="Near")
# sns.regplot(x="delta_theta", y="error_theta", data=df[df["dist"] == 0], x_bins=np.arange(-torch.pi, torch.pi, torch.pi/10), scatter=True, fit_reg=False, label="Far")
# plt.xlabel('distance θ₂ - θ₁ (rad)')
# plt.ylabel('Signed error θ (rad)')
# plt.title('Response Bias Relative to Distractor Stimulus')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("model_analysis_figs/response_bias_relative_to_distractor_stimulus_regression.png")

# # %%

# def circular_mean_window(angles: torch.Tensor) -> torch.Tensor:
#     sin_sum = torch.sin(angles).mean(dim=0)
#     cos_sum = torch.cos(angles).mean(dim=0)
#     return torch.atan2(sin_sum, cos_sum)

# sorted_idx = np.argsort(delta_theta)
# delta_theta_sorted = delta_theta[sorted_idx]
# error_theta_sorted = error_theta[sorted_idx]
# delta_color_sorted = delta_color[sorted_idx]

# # Parameters
# window_width = np.pi / 5   # e.g. window size ~18°
# step_size = np.pi / 400     # e.g. step ~1.8°
# x_vals = np.arange(-np.pi, np.pi, step_size)

# # Compute sliding circular mean
# circ_mean_vals_same = []
# circ_mean_vals_diff = []

# for x in x_vals:
#     # Create sliding window
#     lower = x - window_width / 2
#     upper = x + window_width / 2

#     # Handle circular wraparound (use modular arithmetic)
#     if lower < -np.pi:
#         mask = (delta_theta_sorted >= (lower + 2 * np.pi)) | (delta_theta_sorted < upper)
#     elif upper > np.pi:
#         mask = (delta_theta_sorted >= lower) | (delta_theta_sorted < (upper - 2 * np.pi))
#     else:
#         mask = (delta_theta_sorted >= lower) & (delta_theta_sorted < upper)

#     values_same = error_theta_sorted[mask & (color_near_far[sorted_idx] == 1)]
#     values_diff = error_theta_sorted[mask & (color_near_far[sorted_idx] == 0)]
#     if len(values_same) == 0:
#         circ_mean_vals_same.append(np.nan)
#     else:
#         circ_mean = circular_mean_window(values_same)
#         circ_mean_vals_same.append(circ_mean)

#     if len(values_diff) == 0:
#         circ_mean_vals_diff.append(np.nan)
#     else:
#         circ_mean = circular_mean_window(values_diff)
#         circ_mean_vals_diff.append(circ_mean)

# circ_mean_vals_same = np.array(circ_mean_vals_same)
# circ_mean_vals_diff = np.array(circ_mean_vals_diff)
# # %%
# plt.figure(figsize=(6, 4))

# # Scatter points
# #plt.scatter(delta_theta, error_theta, alpha=0.3, label='Raw data')

# # Circular mean per bin
# plt.plot(x_vals, circ_mean_vals_same, color='blue', linestyle='--', label='y_hat same color (sliding window)')
# plt.plot(x_vals, circ_mean_vals_diff, color='red', linestyle='--', label='y_hat opposite color (sliding window)')
# plt.scatter(delta_theta_sorted, error_theta_sorted, alpha=0.3)
# #add a legend, NOT a zero line
# plt.xlabel('distance θ₂ - θ₁ (rad)')
# plt.ylabel('Signed error θ (rad)')
# plt.title('Sliding Circular Mean of Response Bias')
# plt.legend()
# plt.savefig("model_analysis_figs/sliding_circular_mean_response_bias.png")
# # %%
