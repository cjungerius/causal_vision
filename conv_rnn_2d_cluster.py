import numpy as np
from scipy.special import iv as besselI
import torch
from torch import randn
from torch.distributions import VonMises
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import matplotlib.pyplot as plt
from section_chris.ReluRNNLayer import MyRNN
from section_6.utils import generate_blank_sensory_input, errors_spatial
from skimage.color import lab2rgb
from tqdm import tqdm

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
def my_loss(output, target):
    # Loss function that computes the mean squared error, with target being the angle in radians
    target_x = torch.cos(target).unsqueeze(-1)
    target_y = torch.sin(target).unsqueeze(-1)
    output_x = output[:, :, 0]
    output_y = output[:, :, 1]

    x_errors = (output_x - target_x)**2
    y_errors = (output_y - target_y)**2
    return (x_errors + y_errors).mean()



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

    (x, y) = torch.meshgrid(s, s)

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
        target_colors = torch.rand(batch_size) * 2 * torch.pi
        distractor_angles = torch.rand(batch_size) * 2 * torch.pi
        distractor_colors = torch.rand(batch_size) * 2 * torch.pi
        angles_1 = target_angles
        angles_2 = distractor_angles
        colors_1 = target_colors
        colors_2 = distractor_colors
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
        p_color_diff = p - q * p

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
    
    # Convert to (C, H, W) f    # Create Gabor patches for target and distractor angles
    target_gabors = torch.stack([gabor(232, sigma=.4, theta=angle/2, Lambda=.25, psi=0, gamma=1) for angle in angles_1])
    target_gabors += 1
    target_gabors /= 2
    target_gabors *= 75
    distractor_gabors = torch.stack([gabor(232, sigma=.4, theta=angle/2, Lambda=.25, psi=0, gamma=1) for angle in angles_2])
    distractor_gabors += 1
    distractor_gabors /= 2
    distractor_gabors *= 75
    # Convert to (C, H, W) format
    target_gabors = target_gabors.unsqueeze(1)  # Add channel dimension
    target_gabors = target_gabors.repeat(1, 3, 1, 1)  # Increase color dimensions to 3
    target_gabors[:,1,1,1] = torch.cos(colors_1)
    target_gabors[:,2,1,1] = torch.sin(colors_1)
    target_gabors = lab2rgb(target_gabors.numpy(), channel_axis = 1)
    target_gabors = preprocess(torch.from_numpy(target_gabors))
    distractor_gabors = distractor_gabors.unsqueeze(1)  
    distractor_gabors = distractor_gabors.repeat(1, 3, 1, 1)
    distractor_gabors[:,1,1,1] = torch.cos(colors_2)
    distractor_gabors[:,2,1,1] = torch.sin(colors_2)
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

    return target_features, distractor_features, target_angles, angles_1, angles_2


## Define our system
dt = 0.01
tau = 0.25  # time constant for the RNN dynamics
n_a = 128 # input size
N = 400 
batch_size = 20
rnn = MyRNN(n_a, N, 2, dt, tau)
rnn.to(device)

## Define training machinery
lr = 5e-4  # Reduced learning rate for larger network
opt = torch.optim.Adam(rnn.parameters(), lr)
num_batches = 500
losses = [] # Store for loss curve plotting
dim1_errs = [] # Store for accuracies curve plotting
dim2_errs = []

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

## Initialise simulation, including the first noise term
eta_tilde = torch.randn(batch_size, N)
eta_tilde = eta_tilde.to(device)
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

network_losses = []  # Store for network loss
optimal_losses = []  # Store for optimal loss

for b in tqdm(range(num_batches)):

    opt.zero_grad()

    #flip first and second here to change whether the network should focus on the first or second period (the first argument==the target period)
    second_input, first_input, target_angles, second_angles, first_angles = generate_batch(batch_size, p, kappa_tilde)
    first_input = first_input.to(device)
    second_input = second_input.to(device)
    target_angles = target_angles.to(device)

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

    all_network_outputs = torch.stack(batch_network_outputs, 1)
    loss = my_loss(all_network_outputs, target_angles)
    
    # Compute ideal observer loss for comparison
    ideal_estimates = compute_ideal_observer_estimates(second_angles.to("cpu"), first_angles.to("cpu"), kappa_tilde, p)
    # Convert ideal estimates to 2D output format for loss computation
    ideal_angles = ideal_estimates
    ideal_x = torch.cos(ideal_angles).unsqueeze(1).repeat(1, resp_timesteps).unsqueeze(2)
    ideal_y = torch.sin(ideal_angles).unsqueeze(1).repeat(1, resp_timesteps).unsqueeze(2)
    ideal_outputs = torch.cat([ideal_x, ideal_y], dim=2)
    ideal_loss = my_loss(ideal_outputs, target_angles.to("cpu"))
    
    loss.backward()
    opt.step()

    x_err, y_err = errors_spatial(target_angles.to("cpu"), all_network_outputs.detach().to("cpu"), 2 * torch.pi)

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
        fig.savefig('grid_losses.png')

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
plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')

print(f"Final Network Loss: {network_losses[-1]:.6f}")
print(f"Final Ideal Observer Loss: {optimal_losses[-1]:.6f}")
print(f"Final Performance Gap: {loss_gap[-1]:.6f}")
print(f"Mean Performance Gap: {np.mean(loss_gap):.6f}")
print(f"Std Performance Gap: {np.std(loss_gap):.6f}")

torch.save(rnn.state_dict(), 'cnn_rnn_gabor_2d_model.pth')
