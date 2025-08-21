from abc import ABC, abstractmethod
import torch
from torch import pi
import warnings
from skimage.color import lab2rgb
from torch.distributions import VonMises


class Task(ABC):
    def __init__(self, p, q, kappa_1, kappa_2, n_a):
        self.p = p
        self.q = q
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.n_a = n_a
        self.vm_1 = VonMises(0, self.kappa_1)
        self.vm_2 = VonMises(0, self.kappa_2)

    def generate_angles(self, batch_size):
        return torch.rand(batch_size) * 2 * pi

    @abstractmethod
    def generate_batch(self, batch_size: int) -> dict:
        pass

    @abstractmethod
    def generate_inputs(self, batch) -> tuple:
        pass


class AngleTask1D(Task):
    def __init__(self, p, kappa, n_a):
        super().__init__(p, 0, kappa, 1, n_a)

    def generate_batch(self, batch_size: int):
        targets = self.generate_angles(batch_size)
        distractors = self.generate_angles(batch_size)
        distractors = torch.where(torch.rand(batch_size) < self.p, targets, distractors)

        targets_observed = targets + self.vm_1.sample((batch_size,)) % (2 * pi)
        distractors_observed = distractors + self.vm_1.sample((batch_size,)) % (2 * pi)

        return {
            "targets": targets,
            "distractors": distractors,
            "targets_observed": targets_observed,
            "distractors_observed": distractors_observed,
        }

    def generate_inputs(self, batch):
        target_inputs = bump_input(batch["targets_observed"], self.n_a, 1, self.kappa_1)
        distractor_inputs = bump_input(
            batch["distractors_observed"], self.n_a, 1, self.kappa_1
        )
        return (target_inputs, distractor_inputs)


class AngleTask2D(Task):
    def __init__(self, p, q, kappa_1, kappa_2, n_a):
        super().__init__(p, q, kappa_1, kappa_2, n_a)


class FeatureTask1D(AngleTask1D):
    def __init__(self, p, kappa, n_a):
        super().__init__(p, kappa, n_a)


class FeatureTask2D(AngleTask2D):
    def __init__(self, p, q, kappa_1, kappa_2, n_a):
        super().__init__(p, q, kappa_1, kappa_2, n_a)


def generate_angles(batch_size):
    return torch.rand(batch_size) * 2 * pi


def bump_input(angles, n_a, A, kappa):
    batch_size = angles.shape[0]
    unit_indices = torch.arange(1, n_a + 1).unsqueeze(0).repeat(batch_size, 1)
    angle_diffs = (angles.unsqueeze(1) - unit_indices) * 2 * pi / n_a
    bumps = A * (kappa * angle_diffs.cos()).exp()
    return bumps


def make_gabors(
    size: int,
    sigma: float,
    theta: torch.Tensor,
    Lambda: float,
    psi: float,
    gamma: float,
) -> torch.Tensor:
    """Draw a gabor patch."""
    sigma_x = sigma
    sigma_y = sigma / gamma

    # Bounding box
    s = torch.linspace(-1, 1, size)

    (x, y) = torch.meshgrid(s, s, indexing="xy")

    theta = theta.view(*theta.shape, 1, 1)

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
    gabors = make_gabors(
        size=232, sigma=0.4, theta=angles / 2, Lambda=0.25, psi=0, gamma=1
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
        gabors = lab2rgb(gabors.numpy(), channel_axis=1)

    # Preprocess and extract features
    gabors = preprocess(torch.from_numpy(gabors)).to(device)
    with torch.no_grad():
        features = model.features(gabors)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)

    return features
