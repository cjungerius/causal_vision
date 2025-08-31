import torch
import warnings
from skimage.color import lab2rgb


def my_loss_feature(output, target):
    target = target.unsqueeze(1)
    errors = (output - target) ** 2
    return errors.mean()


def my_loss_spatial(output, target):
    # Loss function that computes the mean squared error, with target being the angle in radians
    target_x = torch.cos(target).unsqueeze(-1)
    target_y = torch.sin(target).unsqueeze(-1)
    output_x = output[:, :, 0]
    output_y = output[:, :, 1]

    x_errors = (output_x - target_x) ** 2
    y_errors = (output_y - target_y) ** 2
    return (x_errors + y_errors).mean()


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
