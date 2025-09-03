import torch
import warnings
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
import numpy as np


def my_loss_feature(target, outputs):
    target = target.unsqueeze(1)
    errors = (outputs - target) ** 2
    return errors.mean()


def my_loss_spatial(target, outputs):
    # Loss function that computes the mean squared error, with target being the angle in radians
    target_x = torch.cos(target).unsqueeze(-1)
    target_y = torch.sin(target).unsqueeze(-1)
    output_x = outputs[:, :, 0]
    output_y = outputs[:, :, 1]

    x_errors = (output_x - target_x) ** 2
    y_errors = (output_y - target_y) ** 2
    return (x_errors + y_errors).mean()


def criterion(batch, params):
    if params.input_type == "spatial" or params.output_type == "angle":
        target = batch["target_angles"]
        outputs = batch["outputs"]
        return my_loss_spatial(target, outputs)
    
    elif params.output_type == "angle_color":
        target_1 = batch["target_angles"]
        target_2 = batch["target_colors"]
        outputs_1 = batch["outputs"][:, :, :2]
        outputs_2 = batch["outputs"][:, :, 2:]
        loss_1 = my_loss_spatial(target_1, outputs_1)
        loss_2 = my_loss_spatial(target_2, outputs_2)
        return loss_1 + loss_2
    
    else:
        target = generate_gabor_features(
            batch["target_angles"],
            batch["target_colors"],
            params.model,
            params.device,
            params.preprocess,
        )
        outputs = batch["outputs"]
        return my_loss_feature(target, outputs)


def _make_gabors(
    size: int,
    sigma: float,
    theta: torch.Tensor,
    Lambda: float,
    psi: float,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """Draw one or a batch of Gabor patches.

    Args:
        size (int): Output height/width in pixels. Grid spans [-1, 1] in both axes.
        sigma (float): Gaussian std along x in grid units. y std is sigma/gamma.
        theta (torch.Tensor): Orientation angle(s) in radians. Can be scalar or any
            shape; treated as batch dims and broadcast over HxW.
        Lambda (float): Wavelength of the sinusoid in grid units
            (e.g., 0.25 ≈ 4 cycles across the width).
        psi (float): Phase offset in radians.
        gamma (float): Spatial aspect ratio. >1 shrinks y (elliptical envelope).

    Returns:
        torch.Tensor: Gabor patch(es) with shape (*theta.shape, size, size).
            Dtype/device follow PyTorch promotion rules between theta and the grid.

    Notes:
        - Uses torch.meshgrid(indexing="xy"); orientation is measured from the x-axis.
        - The grid is created on CPU with default float dtype; pass theta on CPU or
          adapt the function to build the grid on theta.device/dtype.

    Example:
        >>> theta = torch.tensor([0.0, torch.pi/4, torch.pi/2])
        >>> gb = make_gabors(64, 0.4, theta, 0.25, 0.0, 1.0)
        >>> gb.shape
        torch.Size([3, 64, 64])
    """
    sigma_x = sigma
    sigma_y = sigma / gamma

    # Bounding box
    s = torch.linspace(-1, 1, size, device=device)

    (x, y) = torch.meshgrid(s, s, indexing="xy")

    theta = theta.view(*theta.shape, 1, 1)

    # Rotation
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    gb = torch.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * torch.cos(2 * torch.pi / Lambda * x_theta + psi)
    return gb


def generate_gabor_features(angles, colors, model, device, preprocess) -> torch.Tensor:
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
    gabors = gabors.unsqueeze(1)  # Add channel dimension
    gabors = gabors.repeat(1, 3, 1, 1)  # Expand to 3 channels

    # Encode colors in LAB space (L=74, a=cos*37, b=sin*37)
    gabors[:, 1, :, :] = torch.cos(colors).view(*colors.shape, 1, 1) * 37
    gabors[:, 2, :, :] = torch.sin(colors).view(*colors.shape, 1, 1) * 37

    # Convert LAB to RGB with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        gabors = lab2rgb(gabors.cpu().numpy(), channel_axis=1)

    # Preprocess and extract features
    gabors = preprocess(torch.from_numpy(gabors)).to(device)
    with torch.no_grad():
        features = model.features(gabors)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)

    return features

def analyze_test_batch_angles(batch):
    theta = batch['target_angles']
    xy = batch['outputs']
    xy = batch['outputs'].mean(dim=1)
    theta_hat = torch.arctan2(xy[:,1], xy[:,0])

    delta_theta = (batch['target_angles_observed'] - batch['distractor_angles_observed']) % (2*torch.pi)
    delta_theta = torch.where(delta_theta >= torch.pi, delta_theta - 2 * torch.pi, delta_theta)

    delta_ideal = (batch['target_angles_observed'] - batch['ideal_observer_estimates']) % (2 * torch.pi)
    delta_ideal = torch.where(delta_ideal >= torch.pi, delta_ideal - 2 * torch.pi, delta_ideal)

    error_theta = (theta - theta_hat) % (2 * torch.pi)
    error_theta = torch.where(error_theta >= torch.pi, error_theta - 2 * torch.pi, error_theta)

    return {
        "theta": theta,
        "theta_hat": theta_hat,
        "delta_theta": delta_theta,
        "error_theta": error_theta,
        "ideal_observer": delta_ideal
    }

def vizualize_test_output(test_output):
    
    fig, axes = plt.subplots(2, figsize=(10, 12))

    axes[0].scatter(test_output['theta'].cpu(), test_output['theta_hat'].cpu() % (2 * torch.pi), alpha=0.5)
    axes[0].set_xlabel("True Angle (rad)")
    axes[0].set_ylabel("Predicted Angle (rad)")
    axes[0].set_title("Angle Prediction")
    axes[0].axis("equal")

    # calc sliding window circular avg
    x_vals, circ_mean_vals = calc_sliding_window_avg(test_output['delta_theta'], test_output['error_theta'])
    sorted_idx = np.argsort(test_output['delta_theta'].cpu().numpy())

    axes[1].plot(x_vals, circ_mean_vals, label="Circular Avg", color="red", linestyle='--')
    axes[1].plot(test_output['delta_theta'].cpu()[sorted_idx], test_output['ideal_observer'].cpu()[sorted_idx], label="Ideal Observer", color="blue", linestyle='--')
    axes[1].scatter(test_output['delta_theta'].cpu(), test_output['error_theta'].cpu(), alpha=0.5)
    axes[1].set_xlabel("Delta Angle (rad)")
    axes[1].set_ylabel("Error Angle (rad)")
    axes[1].set_title("Angle Error")

    plt.tight_layout()
    plt.show()
    return

def calc_sliding_window_avg(delta_theta, error_theta):

    def circular_mean(angles: torch.Tensor) -> torch.Tensor:
        sin_sum = torch.sin(angles).mean(dim=0)
        cos_sum = torch.cos(angles).mean(dim=0)
        return torch.atan2(sin_sum, cos_sum)


    sorted_idx = np.argsort(delta_theta)
    delta_theta_sorted = delta_theta[sorted_idx]
    error_theta_sorted = error_theta[sorted_idx]

    # Parameters
    window_width = np.pi / 10   # e.g. window size ~18°
    step_size = np.pi / 100     # e.g. step ~1.8°
    x_vals = np.arange(-np.pi, np.pi, step_size)

    # Compute sliding circular mean
    circ_mean_vals = []

    for x in x_vals:
        # Create sliding window
        lower = x - window_width / 2
        upper = x + window_width / 2

        # Handle circular wraparound (use modular arithmetic)
        if lower < -np.pi:
            mask = (delta_theta_sorted >= (lower + 2 * np.pi)) | (delta_theta_sorted < upper)
        elif upper > np.pi:
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
    return x_vals, circ_mean_vals