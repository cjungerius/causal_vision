# %%
import numpy as np
import torch
from torch import randn
from torch import nn
import torch.nn.functional as F
from torch.distributions import VonMises
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import matplotlib.pyplot as plt
from section_chris.ReluRNNLayer import MyRNN
from section_6.utils import generate_blank_sensory_input
from skimage.color import lab2rgb
import warnings
from sklearn.decomposition import PCA
import pickle
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor

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

# %% Define our system
dt = 0.01
tau = 0.25  # time constant for the RNN dynamics
n_a = 128 # input size
N = 600 
batch_size = 100
rnn = MyRNN(n_a, N, n_a, dt, tau)
rnn.to(device)
rnn.init_hidden(batch_size)

try:
    rnn.load_state_dict(torch.load("cnn_rnn_vector_model.pth", map_location=device))
    print("Loaded saved RNN parameters.")
except FileNotFoundError:
    print("No saved RNN parameters found. Starting with random initialization.")

batch_size = 1000
rnn.init_hidden(batch_size)

# %%
def generate_batch(batch_size, p=0.5, kappa=5.0, q=0.0):
    """Generate a batch of Gabor patches."""

    # Von Mises noise distribution
    vm = VonMises(0, kappa*1000)

    # Generate ground truth angles
    target_angles = torch.rand(batch_size) * 2 * torch.pi #torch.linspace(0,torch.pi,batch_size) #torch.rand(batch_size) * 2 * torch.pi
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
    angles_1 = target_angles #(target_angles + noise) % (2 * torch.pi)
    noise = vm.sample((batch_size,))
    colors_1 = target_colors #(target_colors + noise) % (2 * torch.pi)
    # Add noise to the distractor angles
    noise = vm.sample((batch_size,))
    angles_2 = distractor_angles #(distractor_angles + noise) % (2 * torch.pi) 
    noise = vm.sample((batch_size,))
    colors_2 = distractor_colors #(distractor_colors + noise) % (2 * torch.pi)

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
        gabors[:,1,:,:] = 0 #torch.cos(colors).unsqueeze(-1).unsqueeze(-1) * 37
        gabors[:,2,:,:] = 0 #torch.sin(colors).unsqueeze(-1).unsqueeze(-1) * 37
        
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

    return target_features, distractor_features, ground_truth_features, angles_1, angles_2, target_angles   


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
p = 0
q = 0
eps1 = 0.9
eps2 = (1 - eps1**2) ** 0.5
C = .01

## Initialise simulation, including the first noise term
eta_tilde = torch.randn(batch_size, N)
eta_tilde = eta_tilde.to(device)

# %% test!

with torch.no_grad():

    #flip first and second here to change whether the network should focus on the first or second period (the first argument==the target period)
    second_input, first_input, ground_truth, second_angles, first_angles, ground_truth_angles = generate_batch(batch_size, 0, kappa_tilde, 0)
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
    batch_network_voltages = []

    resp_sensory_input = generate_blank_sensory_input(n_a, False, batch_size)[:,:-1].to(device)
    for ts in range(resp_timesteps):
        eta_tilde = (eps1 * eta_tilde) + (eps2 * randn(batch_size, N, device=device))
        eta = eta_tilde * C
        voltage, network_output = rnn.step(resp_sensory_input, eta, return_output=True)
        batch_network_voltages.append(voltage)
        batch_network_outputs.append(network_output)

    all_network_outputs = torch.stack(batch_network_outputs, 1)


# %%
mean_vector = all_network_outputs.mean(1)  # [batch, 2]

theta = second_angles 
theta_2 = first_angles 

print("difference between output and ground truth: %s" % torch.cosine_similarity(mean_vector,ground_truth).mean())
print("difference between output and main input: %s" % torch.cosine_similarity(mean_vector,second_input).mean())
print("difference between output and accessory input: %s" % torch.cosine_similarity(mean_vector,first_input).mean())

# %%
def circular_distance(b1, b2):
    r = (b2 - b1) % (2*torch.pi)
    r = torch.where(r >= torch.pi, r - 2*torch.pi, r)
    return r

# Compute signed angle differences
delta_theta = circular_distance(theta_2, theta)        # Direction θ₂ - θ₁
#delta_theta = torch.abs(delta_theta )
# %%
plt.plot(delta_theta,torch.acos(torch.cosine_similarity(mean_vector, second_input)).cpu(),'o')
plt.savefig("model_analysis_figs/vector_similarity_main.png")
plt.close()
plt.plot(delta_theta,torch.acos(torch.cosine_similarity(mean_vector, first_input)).cpu(),'o')
plt.savefig("model_analysis_figs/vector_similarity_accessory.png")
plt.close()

first_angle = torch.acos(torch.cosine_similarity(mean_vector, first_input)).cpu()
second_angle = torch.acos(torch.cosine_similarity(mean_vector, second_input)).cpu()
total_angle = first_angle + second_angle
plt.plot(delta_theta,first_angle/total_angle,'o')
plt.savefig("model_analysis_figs/vector_comparing_sims.png")
# %%
all_test_trial_outputs = torch.stack(batch_network_outputs, 1).detach().cpu().numpy()
dim_reduced_outputs_flat = PCA(n_components=2).fit_transform(all_test_trial_outputs.reshape(batch_size * resp_timesteps, -1))
dim_reduced_outputs = dim_reduced_outputs_flat.reshape(batch_size, resp_timesteps, -1)

colors = ground_truth_angles.unsqueeze(-1).cpu().numpy() / (torch.pi * 2)
fig, axes = plt.subplots(1)
for i, drv in enumerate(dim_reduced_outputs):
    axes.plot(*drv.T, label = i+1, color=plt.cm.RdYlBu(colors[i])) # type: ignore
    axes.scatter(*drv[0,:,None], marker='o', color=plt.cm.RdYlBu(colors[i])) # type: ignore
    axes.scatter(*drv[-1,:,None], marker='x', color=plt.cm.RdYlBu(colors[i])) # type: ignore
fig.suptitle('PCA of output trajectories. Response period starts at o and ends at x')
fig.savefig('model_analysis_figs/vector_pca_reprs_output.png')
# %%
all_test_trial_voltages = torch.stack(batch_network_voltages, 1).detach().cpu().numpy()
dim_reduced_voltages_flat = PCA(n_components=2).fit_transform(all_test_trial_voltages.reshape(batch_size * resp_timesteps, -1))
dim_reduced_voltages = dim_reduced_voltages_flat.reshape(batch_size, resp_timesteps, -1)

for i, drv in enumerate(dim_reduced_voltages):
    axes.plot(*drv.T, label = i+1, color=plt.cm.RdYlBu(colors[i])) # type: ignore
    axes.scatter(*drv[0,:,None], marker='o', color=plt.cm.RdYlBu(colors[i])) # type: ignore
    axes.scatter(*drv[-1,:,None], marker='x', color=plt.cm.RdYlBu(colors[i])) # type: ignore
fig.suptitle('PCA of voltage trajectories. Response period starts at o and ends at x')
fig.savefig('model_analysis_figs/vector_pca_reprs_voltages.png')

def circular_mean_window(angles: torch.Tensor) -> torch.Tensor:
    sin_sum = torch.sin(angles).mean(dim=0)
    cos_sum = torch.cos(angles).mean(dim=0)
    return torch.atan2(sin_sum, cos_sum)

#error_theta = 1 - torch.cosine_similarity(ground_truth, mean_vector)
error_theta = torch.acos(torch.cosine_similarity(ground_truth, mean_vector)).cpu()

sorted_idx = np.argsort(delta_theta)
delta_theta_sorted = delta_theta[sorted_idx].cpu()
error_theta_sorted = error_theta[sorted_idx].cpu()

# Parameters
window_width = np.pi / 5   # e.g. window size ~18°
step_size = np.pi / 400     # e.g. step ~1.8°
x_vals = np.arange(-np.pi, np.pi, step_size)

# Compute sliding circular mean
circ_mean_vals = []

for x in x_vals:
    # Create sliding window
    lower = x - window_width / 2
    upper = x + window_width / 2

    # Handle circular wraparound (use modular arithmetic)
    if lower < -torch.pi:
        mask = (delta_theta_sorted >= (lower + 2*np.pi)) | (delta_theta_sorted < upper)
    elif upper > np.pi:
        mask = (delta_theta_sorted >= lower) | (delta_theta_sorted < (upper - 2*np.pi))
    else:
        mask = (delta_theta_sorted >= lower) & (delta_theta_sorted < upper)

    values = error_theta_sorted[mask]
    if len(values) == 0:
        circ_mean_vals.append(np.nan)
    else:
        circ_mean = circular_mean_window(values)
        circ_mean_vals.append(circ_mean)

circ_mean_vals = np.array(circ_mean_vals)
# %%
plt.figure(figsize=(6, 4))

# Scatter points
#plt.scatter(delta_theta, error_theta, alpha=0.3, label='Raw data')

# Circular mean per bin
plt.plot(x_vals, circ_mean_vals, color='blue', linestyle='--', label='y_hat (sliding window)')
plt.scatter(delta_theta_sorted, error_theta_sorted, alpha=0.3)
#add a legend, NOT a zero line
plt.xlabel('distance θ₂ - θ₁ (rad)')
plt.ylabel('Angle between response vector and main input vector θ (rad)')
plt.title('Sliding Circular Mean of Response Bias')
plt.legend()
plt.savefig("model_analysis_figs/vector_sliding_circular_mean_response_bias.png")
# %%

class MyModel(nn.Module):
    def __init__(self, input_size, N, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, N)
        self.fc2 = nn.Linear(N,N)
        self.fc3 = nn.Linear(N,output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
nn_decoder = MyModel(128,100,2)
nn_decoder.to(device)

try:
    nn_decoder.load_state_dict(torch.load("vector_angle_decoder_nn.pth", map_location=device))
    print("Loaded saved NN decoder parameters.")
except FileNotFoundError:
    print("No saved NN decoder parameters found. Starting with random initialization.")


with open('vector_angle_decoder_svm.pkl','rb') as f:
    svm_decoder = pickle.load(f)

#validation = svm_decoder.predict(ground_truth.cpu())
validation = nn_decoder(ground_truth).detach().cpu().numpy()
val_angle = np.arctan2(validation[:,1], validation[:,0]) % (2 * np.pi)

#prediction = svm_decoder.predict(mean_vector.cpu())
prediction = nn_decoder(mean_vector).detach().cpu()
angle_predicted = np.arctan2(prediction[:, 1], prediction[:, 0]) % (2*np.pi) # type: ignore

# Compute sliding circular mean
circ_mean_vals = []
error_theta_sorted = circular_distance(ground_truth_angles.cpu(),torch.tensor(angle_predicted))[sorted_idx]
for x in x_vals:
    # Create sliding window
    lower = x - window_width / 2
    upper = x + window_width / 2

    # Handle circular wraparound (use modular arithmetic)
    if lower < -torch.pi:
        mask = (delta_theta_sorted >= (lower + 2*np.pi)) | (delta_theta_sorted < upper)
    elif upper > np.pi:
        mask = (delta_theta_sorted >= lower) | (delta_theta_sorted < (upper - 2*np.pi))
    else:
        mask = (delta_theta_sorted >= lower) & (delta_theta_sorted < upper)

    values = error_theta_sorted[mask]
    if len(values) == 0:
        circ_mean_vals.append(np.nan)
    else:
        circ_mean = circular_mean_window(values)
        circ_mean_vals.append(circ_mean)

circ_mean_vals = np.array(circ_mean_vals)

f, ax = plt.subplots(ncols=1,nrows=3,figsize=(6,10))
plt.tight_layout(h_pad=5)
ax[0].scatter(ground_truth_angles.cpu(), val_angle)
ax[0].set(ylabel="y_hat (rad)", xlabel="y (rad)",title="NN decoding of ground truth inputs")
ax[1].scatter(ground_truth_angles.cpu(), angle_predicted)
ax[1].set(ylabel="y_hat (rad)", xlabel="y (rad)",title="NN decoding of model outputs")
ax[2].scatter(delta_theta.cpu(), circular_distance(ground_truth_angles.cpu(),torch.tensor(angle_predicted)))
ax[2].plot(x_vals, circ_mean_vals,color='blue', linestyle='--')
ax[2].set(ylabel="delta y_hat (rad)", xlabel="delta_theta (rad)",title="circular distance between decoded output and ground truth as a function of delta_theta")

plt.savefig("model_analysis_figs/nn_decoding_analysis.png")
# %%
