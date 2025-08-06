# %%

import numpy as np
import torch
from torch import randn
from torch.nn import MSELoss
from torch.distributions import VonMises
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import matplotlib.pyplot as plt
from section_chris.ReluRNNLayer import MyRNN
from section_6.utils import generate_blank_sensory_input
from skimage.color import lab2rgb
import warnings
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
#my_loss = MSELoss(reduction="mean")
def my_loss(output, target):
    target = target.unsqueeze(1)
    errors = (output - target)**2
    return errors.mean()


# %%
def generate_batch(batch_size, p=0.5, kappa=5.0, q=0.0):
    """Generate a batch of Gabor patches."""

    # Von Mises noise distribution
    vm = VonMises(0, kappa)

    # Generate ground truth angles
    target_angles = torch.rand(batch_size) * 2 * torch.pi
    target_colors = torch.rand(batch_size) * 2 * torch.pi
    # Generate distractor angles
    distractor_angles = torch.rand(batch_size) * 2 * torch.pi
    distractor_colors = torch.rand(batch_size) * 2 * torch.pi

    probability_vector = torch.tensor([
        p * p + q * np.sqrt(p ** 2 * (1 - p) ** 2),
        p * (1 - p) - q * np.sqrt(p ** 2 * (1 - p) ** 2),
        (1 - p) * p - q * np.sqrt(p ** 2 * (1 - p) ** 2),
        (1 - p) ** 2 + q * np.sqrt(p ** 2 * (1 - p) ** 2)
    ])

    class_vector = probability_vector.multinomial(batch_size, replacement=True)
    
    distractor_angles = torch.where(torch.logical_or(class_vector == 0, class_vector == 1) , target_angles, distractor_angles)
    distractor_colors = torch.where(torch.logical_or(class_vector == 0, class_vector == 2), target_colors, distractor_colors)

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
    
    def generate_gabor_features(angles, colors, model, device, preprocess):
        """
        Generate Gabor patches with color encoding and extract features using a CNN model.
        
        Args:
            angles (torch.Tensor): Tensor of angles for Gabor orientation (in radians)
            colors (torch.Tensor): Tensor of color angles (in radians)
            model: CNN model for feature extraction
            device: Target device (e.g., 'cuda')
            preprocess: Image preprocessing function
            
        Returns:
            torch.Tensor: Extracted features (flattened)
        """
        # Generate Gabor patches
        gabors = torch.stack([gabor(232, sigma=.4, theta=angle/2, Lambda=.25, psi=0, gamma=1) 
                            for angle in angles])
        # Normalize to [0, 74] range
        gabors = (gabors + 1) / 2 * 74
        
        # Convert to (C, H, W) format and add color channels
        gabors = gabors.unsqueeze(1)  # Add channel dimension
        gabors = gabors.repeat(1, 3, 1, 1)  # Expand to 3 channels
        
        # Encode colors in LAB space (L=74, a=cos*37, b=sin*37)
        gabors[:,1,:,:] = torch.cos(colors).unsqueeze(-1).unsqueeze(-1) * 37
        gabors[:,2,:,:] = torch.sin(colors).unsqueeze(-1).unsqueeze(-1) * 37
        
        # Convert LAB to RGB with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            gabors = lab2rgb(gabors.numpy(), channel_axis=1)
        
        # Preprocess and extract features
        gabors = preprocess(torch.from_numpy(gabors)).to(device)
        with torch.no_grad():
            features = model.features(gabors)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
        
        return features

    ground_truth_features = generate_gabor_features(target_angles, target_colors, model, device, preprocess)
    target_features = generate_gabor_features(angles_1, colors_1, model, device, preprocess)
    distractor_features = generate_gabor_features(angles_2, colors_2, model, device, preprocess)

    return target_features, distractor_features, ground_truth_features   


# %% Define our system
dt = 0.01
tau = 0.25  # time constant for the RNN dynamics
n_a = 128 # input size
N = 600 
batch_size = 20
rnn = MyRNN(n_a, N, n_a, dt, tau)
rnn.to(device)

## Define training machinery
lr = 0.001 
opt = torch.optim.Adam(rnn.parameters(), lr)
num_batches = 2000
losses = [] # Store for loss curve plotting

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
q = 3/5
eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = .01

## Initialise simulation, including the first noise term
eta_tilde = torch.randn(batch_size, N)
eta_tilde = eta_tilde.to(device)

network_losses = []  # Store for network loss


# %% train!

for b in tqdm(range(num_batches)):

    opt.zero_grad()

    #flip first and second here to change whether the network should focus on the first or second period (the first argument==the target period)
    second_input, first_input, ground_truth = generate_batch(batch_size, p, kappa_tilde, q)
    first_input = first_input.to(device)
    second_input = second_input.to(device)
    ground_truth = ground_truth.to(device)

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
    # output_mean = all_network_outputs.mean(dim=1)
    loss = my_loss(all_network_outputs, ground_truth)

    loss.backward()
    opt.step()

    losses.append(loss.item())
    network_losses.append(loss.item())

    if b % 200 == 0:
        plt.close('all')

        plt.plot(losses, label='Network Loss')
        plt.legend()

        print(f"Batch {b}: Network Loss: {losses[-1]:.4f}")
        plt.savefig('vector_losses.png')


torch.save(rnn.state_dict(), 'cnn_rnn_vector_model_local.pth')


# %%
