# %%

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np 
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
import warnings
from skimage.color import lab2rgb
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def generate_batch(batch_size):
    """Generate a batch of Gabor patches."""

    # Generate ground truth angles
    labels = torch.rand(batch_size) * 2 * torch.pi
        
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
    
    def generate_gabor_features(angles, model, device, preprocess):
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
        gabors[:,1,:,:] = 0
        gabors[:,2,:,:] = 0        
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

    features = generate_gabor_features(labels, model, device, preprocess)

    return features, labels

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
my_model = MyModel(128,100,2)
my_model.to(device)

# %%
# Train the model for a few epochs
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
batch_size = 10
num_batches = 300  
model_losses = []

def my_loss(output, target):
    # Loss function that computes the mean squared error, with target being the angle in radians
    target_x = torch.cos(target)
    target_y = torch.sin(target)
    output_x = output[:, 0]
    output_y = output[:, 1]

    x_errors = (output_x - target_x)**2
    y_errors = (output_y - target_y)**2
    return (x_errors + y_errors).mean()


# %%

for b in tqdm(range(num_batches)):
    optimizer.zero_grad()
    features, labels = generate_batch(batch_size)
    features = features.to(device)
    labels = labels.to(device)
    output = my_model(features)
    loss = my_loss(output, labels)
    loss.backward()
    optimizer.step()

    model_losses.append(loss.item())

    if (b + 1) % 100 == 0:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot 1: Loss progression
        ax.plot(model_losses, 'b-', label='Model Loss', linewidth=2)
        ax.set_xlabel('Batch Number')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('decoding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

plt.close('all')

print(f"\nTraining completed! Final losses:")
print(f"Model Loss: {model_losses[-1]:.4f}")


# %%
X_train, y_train = generate_batch(250)
X_test, y_test = generate_batch(50)

# %%
y_train_multi = np.array([np.cos(y_train.cpu()), np.sin(y_train.cpu())]).T
y_test_multi = np.array([np.cos(y_test.cpu()), np.sin(y_test.cpu())]).T

regressor = MultiOutputRegressor(svm.SVR(kernel='rbf', C=1e3, gamma=0.1))

regressor.fit(X_train.cpu(), y_train_multi)
prediction = regressor.predict(X_test.cpu())

angle_predicted = np.arctan2(prediction[:, 1], prediction[:, 0]) % (2*np.pi) # type: ignore

f, ax = plt.subplots(figsize=(4, 3))
ax.scatter(y_test, angle_predicted, lw=0)
ax.set(ylabel="Predicted $y$", xlabel="Test $y$")
# %%
# comparing SVM and NN approach:

y_hat_svm = angle_predicted
model_output = my_model(X_test).detach().cpu()
y_hat_nn = torch.arctan2(model_output[:,1], model_output[:,0]) % (2 * torch.pi)
f, ax = plt.subplots(figsize=(4, 3))
#ax.scatter(y_test, y_hat_svm, alpha=.3)
ax.scatter(y_test, y_test-y_hat_nn, alpha=.3, c="orange")

# %%