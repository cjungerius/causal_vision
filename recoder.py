import torch
from torch import nn
import torch.nn.functional as F
import warnings
from skimage.color import lab2rgb
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ConvNeXt_Base_Weights.DEFAULT
model = convnext_base(weights=weights)
model.to(device)
model.eval()
# Use only first two layers for feature extraction
model.features = torch.nn.Sequential(*list(model.features.children())[:2])
for p in model.features.parameters():
    p.requires_grad = False
model = model
preprocess = weights.transforms()


def _make_gabors(
    size: int,
    sigma: float,
    theta: torch.Tensor,
    Lambda: float,
    psi: float,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    sigma_x = sigma
    sigma_y = sigma / gamma

    # Bounding box
    s = torch.linspace(-1, 1, size, device=device)

    (x, y) = torch.meshgrid(s, s, indexing="ij")

    theta = theta.view(*theta.shape, 1, 1)

    # Rotation
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    gb = torch.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * torch.cos(2 * torch.pi / Lambda * x_theta + psi)
    return gb


def generate_gabor_features(angles, colors, model, device, preprocess):
    # Generate Gabor patches
    gabors = _make_gabors(
        size=232,
        sigma=0.4,
        theta=angles / 2,
        Lambda=0.25,
        psi=0,
        gamma=1,
        device=device,
    )
    # Normalize to [0, 74] range
    gabors = (gabors + 1) / 2 * 74

    # Convert to (C, H, W) format and add color channels
    gabors = gabors.unsqueeze(1).repeat(1, 3, 1, 1)

    # Encode LAB
    if colors is None:
        gabors[:, 1, :, :] = 0
        gabors[:, 2, :, :] = 0
    else:
        gabors[:, 1, :, :] = torch.cos(colors).view(*colors.shape, 1, 1) * 37
        gabors[:, 2, :, :] = torch.sin(colors).view(*colors.shape, 1, 1) * 37
    # Convert LAB to RGB with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        gabors = torch.from_numpy(lab2rgb(gabors.cpu().numpy(), channel_axis=1))

    # Preprocess and extract features
    processed_gabors = preprocess(gabors).to(device)
    with torch.no_grad():
        features = model.features(processed_gabors)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)

    return features, gabors


class MyModel(nn.Module):
    def __init__(self, input_size, output_channels=3):
        super().__init__()

        # 1. Project the latent vector (input_size) to a spatial map (64 channels x 29 x 29)
        # We choose 64 channels as a balance between quality and size.
        self.initial_channels = 64
        self.initial_size = 29
        self.fc_projection = nn.Linear(
            input_size, self.initial_channels * self.initial_size * self.initial_size
        )

        # 2. Upsampling layers
        # Each layer doubles the spatial dimension (29 -> 58 -> 116 -> 232)
        # Kernel=4, Stride=2, Padding=1 is the standard "double size" configuration.
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        # Final layer to reach 232x232 and standard RGB range
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Force output to be between [0, 1] for image display
        )

    def forward(self, x):
        # x shape: (Batch_Size, 128)

        # 1. Expand to linear features
        x = self.fc_projection(x)

        # 2. Reshape into a 3D volume (Batch, Channels, Height, Width)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)

        # 3. Apply upsampling
        x = self.layer1(x)  # -> 58x58
        x = self.layer2(x)  # -> 116x116
        x = self.layer3(x)  # -> 232x232

        return x


# %%
my_model = MyModel(128)
my_model.to(device)

# --- 1. Setup Training Hyperparameters ---
BATCH_SIZE = 100
NUM_EPOCHS = 3000  # Adjust as needed
LEARNING_RATE = 1e-3  # Slightly lower LR usually stabilizes reconstruction tasks

# Re-initialize optimizer with new LR if desired
optimizer = torch.optim.Adam(my_model.parameters(), lr=LEARNING_RATE)

# Use Mean Squared Error for reconstruction
criterion = nn.MSELoss()

print(f"Training on device: {device}")

# --- 2. Training Loop ---
loss_history = []

for epoch in tqdm(range(NUM_EPOCHS)):
    my_model.train()  # Set model to training mode

    # --- A. Generate Random Training Data on the Fly ---
    # We generate random angles [0, 2pi] and random colors
    random_angles = torch.rand(BATCH_SIZE, device=device) * 2 * torch.pi
    random_colors = torch.rand(BATCH_SIZE, device=device) * 2 * torch.pi

    # --- B. Get Ground Truth and Features ---
    # We use your existing function.
    # features: The input to your MyModel (128,)
    # gt_images: The target output (3, 232, 232) - Note: these are preprocessed (normalized)
    features, gt_images = generate_gabor_features(
        random_angles, random_colors, model, device, preprocess
    )

    # Ensure features are float32 (sometimes numpy conversion leaves them as double)
    features = features.float()
    gt_images = gt_images.to(device)

    # --- C. Forward Pass ---
    # 1. Pass features through your model
    reconstructed_img = my_model(features)

    # --- D. Calculate Loss ---
    loss = criterion(reconstructed_img, gt_images)

    # --- E. Backward Pass ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        tqdm.write(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}")

print("Training Complete.")

# --- Visualization ---
my_model.eval()
with torch.no_grad():
    # Generate 1 test sample
    test_angle = torch.tensor([0.0], device=device)  # Vertical
    test_color = torch.tensor([0.0], device=device)

    feat, img_gt = generate_gabor_features(
        test_angle, test_color, model, device, preprocess
    )

    # Reconstruct
    rec_img = my_model(feat.float()).squeeze().permute(1, 2, 0).to("cpu")

    # Reshape back to image dimensions: (C, H, W)
    img_gt = img_gt[0].permute(1, 2, 0).to("cpu")

    img_gt_vis = np.clip(img_gt.numpy(), 0, 1)
    rec_img_vis = np.clip(rec_img.numpy(), 0, 1)

    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    ax[0, 0].imshow(img_gt_vis)
    ax[0, 0].set_title("Original (Input to CNN)")
    ax[1, 0].imshow(rec_img_vis)
    ax[1, 0].set_title("Reconstructed (Output of MyModel)")

    # Generate 1 test sample
    test_angle = torch.tensor([torch.pi / 2], device=device)  # Vertical
    test_color = torch.tensor([torch.pi / 2], device=device)

    feat, img_gt = generate_gabor_features(
        test_angle, test_color, model, device, preprocess
    )

    # Reconstruct
    rec_img = my_model(feat.float()).squeeze().permute(1, 2, 0).to("cpu")

    # Reshape back to image dimensions: (C, H, W)
    img_gt = img_gt[0].permute(1, 2, 0).to("cpu")

    img_gt_vis = np.clip(img_gt.numpy(), 0, 1)
    rec_img_vis = np.clip(rec_img.numpy(), 0, 1)
    ax[0, 1].imshow(img_gt_vis)
    ax[0, 1].set_title("Original (Input to CNN)")
    ax[1, 1].imshow(rec_img_vis)
    ax[1, 1].set_title("Reconstructed (Output of MyModel)")

    # Generate 1 test sample
    test_angle = torch.tensor([torch.pi], device=device)  # Vertical
    test_color = torch.tensor([torch.pi], device=device)

    feat, img_gt = generate_gabor_features(
        test_angle, test_color, model, device, preprocess
    )

    # Reconstruct
    rec_img = my_model(feat.float()).squeeze().permute(1, 2, 0).to("cpu")

    # Reshape back to image dimensions: (C, H, W)
    img_gt = img_gt[0].permute(1, 2, 0).to("cpu")

    img_gt_vis = np.clip(img_gt.numpy(), 0, 1)
    rec_img_vis = np.clip(rec_img.numpy(), 0, 1)
    ax[0, 2].imshow(img_gt_vis)
    ax[0, 2].set_title("Original (Input to CNN)")
    ax[1, 2].imshow(rec_img_vis)
    ax[1, 2].set_title("Reconstructed (Output of MyModel)")
    plt.savefig("recoder_output.png")

torch.save(my_model.state_dict(), "feature_to_gabor_recoder_conv.pth")
