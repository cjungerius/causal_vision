# In[1]
import numpy as np
from scipy.special import iv as besselI
import torch
import torch.nn.functional as F
from torch.distributions import VonMises
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from tqdm import tqdm
# %%
# Load the ConvNeXt model with pretrained weights
model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
# Set the model to evaluation mode
model.eval()
weights = ConvNeXt_Base_Weights.DEFAULT
preprocess = weights.transforms()
model.features = model.features[0:2]  # Use only the first two layers for feature extraction


# %%
def my_loss(output, target):
    # Loss function that computes the mean squared error, with target being the angle in radians
    target_x = torch.cos(2 * target)
    target_y = torch.sin(2 * target)
    output_x = output[:, 0]
    output_y = output[:, 1]

    x_errors = (output_x - target_x)**2
    y_errors = (output_y - target_y)**2
    return (x_errors + y_errors).mean()



def gabor(size, sigma, theta, Lambda, psi, gamma):
    """Draw a gabor patch."""
    sigma_x = sigma
    sigma_y = sigma / gamma
    Lambda = torch.tensor(Lambda)
    psi = torch.tensor(psi)
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

def generate_batch(batch_size, p=0.5, kappa=5.0):
    """Generate a batch of Gabor patches."""

    # Von Mises noise distribution
    vm = VonMises(0, kappa)

    # Generate ground truth angles uniformly distributed in [0, pi]
    target_angles = torch.rand(batch_size) * torch.pi

    # Generate distractor angles uniformly distributed in [0, pi]
    distractor_angles = torch.rand(batch_size) * torch.pi
    distractor_angles = torch.where(
        torch.rand(batch_size) < p,
        target_angles,  # If the random number is less than p, use the target angle
        distractor_angles  # Otherwise, use a new random angle
    )

    # Add noise to the target angles
    noise = vm.sample((batch_size,)) / 2
    angles_1 = (target_angles + noise)

    # Add noise to the distractor angles
    noise = vm.sample((batch_size,)) / 2
    angles_2 = (distractor_angles + noise)
    
    # Create Gabor patches for target and distractor angles
    target_gabors = torch.stack([gabor(232, sigma=.4, theta=angle, Lambda=.1, psi=0, gamma=1) for angle in angles_1])
    distractor_gabors = torch.stack([gabor(232, sigma=.4, theta=angle, Lambda=.25, psi=0, gamma=1.5) for angle in angles_2])
    # Convert to (C, H, W) format
    target_gabors = target_gabors.unsqueeze(1)  # Add channel dimension
    target_gabors = target_gabors.repeat(1, 3, 1, 1)  # Increase color dimensions to 3
    target_gabors = preprocess(target_gabors)
    distractor_gabors = distractor_gabors.unsqueeze(1)  # Add channel dimension
    distractor_gabors = distractor_gabors.repeat(1, 3, 1, 1)  # Increase color dimensions to 3
    distractor_gabors = preprocess(distractor_gabors)

    # Forward pass to get the activations for the batch
    with torch.no_grad():
        target_features = model.features[0:2](target_gabors)
        distractor_features = model.features[0:2](distractor_gabors)
        target_features = model.avgpool(target_features)
        distractor_features = model.avgpool(distractor_features)
        target_features = torch.flatten(target_features, 1)
        distractor_features = torch.flatten(distractor_features, 1)

    # Pair target and distractor features correctly
    features = torch.cat((target_features, distractor_features), dim=1)
    
    return features, target_angles, angles_1, angles_2

# %%
input, labels, _, _ = generate_batch(10)
# %%
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    
# %%
my_model = MyModel()
output = my_model(input)

# %%
# Train the model for a few epochs
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
batch_size = 32
num_batches = 1000  # Number of batches to train on
p = 4/6  # Probability of distractor being the same as target
kappa = 5.0  # Concentration parameter for Von Mises distribution

# %%
for b in tqdm(range(num_batches)):
    optimizer.zero_grad()
    input, labels, _, _ = generate_batch(batch_size, p, kappa)
    output = my_model(input)
    loss = my_loss(output, labels)
    loss.backward()
    optimizer.step()

    if (b + 1) % 100 == 0:
        print(f'Epoch {b+1}, Loss: {loss.item()}')

# %%
# Test the model
with torch.no_grad():
    test_batch_size = 1000
    chunk_size = 100
    input_chunk = torch.zeros(chunk_size, 256)  # Adjusted for the model's input size
    labels = torch.zeros(test_batch_size)
    angles_1 = torch.zeros(test_batch_size)
    angles_2 = torch.zeros(test_batch_size)
    output = torch.zeros(test_batch_size, 2)
    for i in range(0, test_batch_size, chunk_size):
        input_chunk, label_chunk, angle_1_chunk, angle_2_chunk = generate_batch(chunk_size, p=0, kappa=1000.0)
        output[i:i+chunk_size] = my_model(input_chunk)
        labels[i:i+chunk_size] = label_chunk
        angles_1[i:i+chunk_size] = angle_1_chunk
        angles_2[i:i+chunk_size] = angle_2_chunk

    loss = my_loss(output, labels)
    print(f'Test Loss: {loss.item()}')

# %%
# Visualize the output
import matplotlib.pyplot as plt
output_angles = torch.atan2(output[:, 1], output[:, 0]) % (2 * torch.pi) / 2
plt.figure(figsize=(10, 5))
plt.scatter(labels.numpy(), output_angles.numpy(), alpha=0.5)
plt.plot([0, torch.pi], [0, torch.pi], color='red', linestyle='--')  # Diagonal line
plt.xlabel('True Angles (radians)')
plt.ylabel('Predicted Angles (radians)')
plt.title('True vs Predicted Angles')


# %%
# Plot angle bias as a function of the distance between target and distractor angles

def circular_distance(b1, b2):
    r = (b2 - b1) % (torch.pi)
    r = torch.where(r >= (torch.pi / 2), r - torch.pi, r)
    return r



angle_diffs = circular_distance(angles_2, angles_1)
angle_error = circular_distance(output_angles, labels)
angle_error = torch.where(angle_diffs < 0, - angle_error, angle_error)  # Ensure positive error for positive angle differences
angle_diffs = torch.abs(angle_diffs)

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

def func_scalar(x, kappa, p):
        p_shared = compute_p_shared(0, x, kappa, p)
        kappa_eff = 2 * kappa * np.cos((0 - x) / 2)
        result = calc_circular_mean_stable(p_shared, kappa_eff, kappa, x / 2, 0)
        return result

func = np.vectorize(func_scalar)

xdata = angle_diffs.numpy()*2
ydata = angle_error.numpy()*2
# Fit the function to the data
from scipy.optimize import curve_fit

popt, pcov = curve_fit(func, xdata, ydata, p0=[kappa, p], bounds=([0, 0], [np.inf, 1]))


# %%
plt.figure(figsize=(10, 5))
plt.scatter(xdata/2, ydata, alpha=0.5)
plt.plot(xdata[np.argsort(xdata)]/2, func(xdata[np.argsort(xdata)], *popt), 'r-', label='Fitted function, kappa={:.2f}, p={:.2f}'.format(popt[0], popt[1]))
plt.plot(xdata[np.argsort(xdata)]/2, func(xdata[np.argsort(xdata)], kappa, p), 'b--', label='Posterior Circular Mean (kappa={:.2f}, p={:.2f})'.format(kappa, p))
plt.xlabel('Angle Difference (radians)')
plt.ylabel('Prediction Error (radians)')
plt.legend()

# %%
