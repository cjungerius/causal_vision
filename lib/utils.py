import torch
from torch import nn
import torch.nn.functional as F
import warnings
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
import numpy as np
import pickle
from .save import to_cpu
from matplotlib.colors import TwoSlopeNorm


def circular_distance(angle1, angle2):
    """Compute the minimum distance between two angles in radians."""
    delta = (angle1 - angle2) % (2 * torch.pi)
    delta = torch.where(delta >= torch.pi, delta - 2 * torch.pi, delta)
    return delta


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


def my_loss_decoded(target, outputs):
    target_x = torch.cos(target).unsqueeze(-1)
    target_y = torch.sin(target).unsqueeze(-1)
    output_x = outputs[:, 0]
    output_y = outputs[:, 1]

    x_errors = (output_x - target_x) ** 2
    y_errors = (output_y - target_y) ** 2
    return (x_errors + y_errors).mean()


def criterion(batch, params):
    if params.output_type == "angle":
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

    elif params.output_type == "feature":
        target = generate_gabor_features(
            batch["target_angles"],
            batch.get("target_colors", None),
            params.model,
            params.device,
            params.preprocess,
        )
        outputs = batch["outputs"]
        return my_loss_feature(target, outputs)

    elif params.output_type == "feature_decode":
        outputs = decoder(
            batch["outputs"].mean(dim=1),
            use_nn_decoder=True,
            device=params.device,
            dims=params.dims,
            need_grad=True,
        )
        target = batch["target_angles"]
        loss = my_loss_decoded(target, outputs[:, :2])
        if params.dims == 2:
            target = batch["target_colors"]
            loss += my_loss_decoded(target, outputs[:, 2:])
        return loss


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

    (x, y) = torch.meshgrid(s, s, indexing="ij")

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
        gabors = lab2rgb(gabors.cpu().numpy(), channel_axis=1)

    # Preprocess and extract features
    gabors = preprocess(torch.from_numpy(gabors)).to(device)
    with torch.no_grad():
        features = model.features(gabors)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)

    return features


def analyze_test_batch(output):
    batch = output["test_batch"]
    theta = batch["target_angles"].cpu()
    color_hat = None

    # check if we're working with a batch from an angle or feature network
    if output["params"].output_type == "angle":
        xy = batch["outputs"].mean(dim=1)
    elif output["params"].output_type == "angle_color":
        xy = batch["outputs"].mean(dim=1)
    elif (
        output["params"].output_type == "feature"
        or output["params"].output_type == "feature_decode"
    ):
        xy = decoder(
            batch["outputs"].mean(dim=1),
            use_nn_decoder=True,
            device=output["params"].device,
            dims=output["params"].dims,
        )

    if (
        output["params"].output_type == "angle_color"
        or (output["params"].output_type == "feature" and output["params"].dims == 2)
        or (
            output["params"].output_type == "feature_decode"
            and output["params"].dims == 2
        )
    ):
        color_hat = torch.arctan2(xy[:, 3], xy[:, 2]).cpu()

    theta_hat = torch.arctan2(xy[:, 1], xy[:, 0]).cpu()

    delta_theta = circular_distance(
        batch["target_angles_observed"], batch["distractor_angles_observed"]
    ).cpu()
    delta_ideal = circular_distance(
        batch["target_angles_observed"], batch["ideal_observer_estimates"]
    ).cpu()

    error_theta = (theta - theta_hat) % (2 * torch.pi)
    error_theta = torch.where(
        error_theta >= torch.pi, error_theta - 2 * torch.pi, error_theta
    )

    if output["params"].dims == 2:
        delta_color = circular_distance(
            batch["target_colors_observed"], batch["distractor_colors_observed"]
        )

        if color_hat is not None:
            color = batch["target_colors"].cpu()
            error_color = (color - color_hat) % (2 * torch.pi)
            error_color = torch.where(
                error_color >= torch.pi, error_color - 2 * torch.pi, error_color
            )

    test_output = {
        "theta": theta,
        "theta_hat": theta_hat,
        "delta_theta": delta_theta,
        "error_theta": error_theta,
        "ideal_observer": delta_ideal,
    }
    if output["params"].dims == 2:
        test_output["delta_color"] = delta_color

        if color_hat is not None:
            test_output["color"] = color
            test_output["color_hat"] = color_hat
            test_output["error_color"] = error_color

    return test_output


def decoder(features, use_nn_decoder=False, device="cpu", dims=1, need_grad=False):
    class MyModel(nn.Module):
        def __init__(self, input_size, N, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, N)
            self.fc2 = nn.Linear(N, N)
            self.fc3 = nn.Linear(N, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    if dims == 1:
        nn_decoder = MyModel(128, 100, 2)
        nn_decoder.to(device)

        try:
            nn_decoder.load_state_dict(
                torch.load("decoders/vector_angle_decoder_nn.pth", map_location=device)
            )
        except FileNotFoundError:
            print(
                "No saved NN decoder parameters found. Starting with random initialization."
            )

        # load SVM decoder from pickle:
        if not use_nn_decoder:
            with open("decoders/vector_angle_decoder_svm.pkl", "rb") as f:
                svm_decoder = pickle.load(f)

    elif dims == 2:
        nn_decoder = MyModel(128, 100, 4)
        nn_decoder.to(device)

        try:
            nn_decoder.load_state_dict(
                torch.load(
                    "decoders/vector_angle_color_decoder_nn.pth", map_location=device
                )
            )
        except FileNotFoundError:
            print(
                "No saved NN decoder parameters found. Starting with random initialization."
            )

        # load SVM decoder from pickle:
        if not use_nn_decoder:
            with open("decoders/vector_angle_color_decoder_svm.pkl", "rb") as f:
                svm_decoder = pickle.load(f)

    if not use_nn_decoder:
        prediction = svm_decoder.predict(features.cpu())
    else:
        if need_grad:
            prediction = nn_decoder(features.to(device))
        else:
            prediction = nn_decoder(features.to(device)).detach().cpu()

    return prediction


def visualize_test_output(test_output, dims=1, fname=None):
    if dims == 1:
        fig, axes = plt.subplots(2, figsize=(10, 12))
    elif "color_hat" in test_output:
        fig, axes = plt.subplots(5, figsize=(10, 12))
    else:
        fig, axes = plt.subplots(3, figsize=(10, 12))

    # Move to CPU and detach
    test_output = to_cpu(test_output)

    # Convert to NumPy for Matplotlib
    theta_np = test_output["theta"].numpy()
    theta_hat_np = (test_output["theta_hat"] % (2 * torch.pi)).numpy()

    axes[0].scatter(theta_np, theta_hat_np, alpha=0.5)
    axes[0].set_xlabel("True Angle (rad)")
    axes[0].set_ylabel("Predicted Angle (rad)")
    axes[0].set_title("Angle Prediction")
    axes[0].axis("equal")

    if dims == 1:
        # Sliding window circular avg (returns NumPy)
        x_vals, circ_mean_vals = calc_sliding_window_avg(
            test_output["delta_theta"], test_output["error_theta"]
        )

        delta_np = test_output["delta_theta"].numpy()
        ideal_np = test_output["ideal_observer"].numpy()
        err_np = test_output["error_theta"].numpy()
        sorted_idx = np.argsort(delta_np)

        axes[1].plot(
            x_vals, circ_mean_vals, label="Circular Avg", color="red", linestyle="--"
        )
        axes[1].plot(
            delta_np[sorted_idx],
            ideal_np[sorted_idx],
            label="Ideal Observer",
            color="blue",
            linestyle="--",
        )
        axes[1].scatter(delta_np, err_np, alpha=0.5)
        axes[1].set_xlabel("Delta Angle (rad)")
        axes[1].set_ylabel("Error Angle (rad)")
        axes[1].set_title("Angle Error")

    elif dims == 2:
        delta_theta_np = test_output["delta_theta"].numpy()
        delta_color_np = test_output["delta_color"].numpy()
        err_np = test_output["error_theta"].numpy()
        ideal_np = test_output["ideal_observer"].numpy()

        # Weighted sums and counts
        err_sum, xbins, ybins = np.histogram2d(
            delta_color_np, delta_theta_np, weights=err_np, bins=(40, 40)
        )
        ideal_sum, _, _ = np.histogram2d(
            delta_color_np, delta_theta_np, weights=ideal_np, bins=(xbins, ybins)
        )
        counts, _, _ = np.histogram2d(
            delta_color_np, delta_theta_np, bins=(xbins, ybins)
        )

        # Mean signed error per bin (mask empty bins)
        err_mean = np.divide(
            err_sum, counts, out=np.full_like(err_sum, np.nan), where=counts > 0
        )
        ideal_mean = np.divide(
            ideal_sum, counts, out=np.full_like(ideal_sum, np.nan), where=counts > 0
        )

        # Symmetric robust scale centered at 0
        finite_abs = np.concatenate(
            [
                np.abs(err_mean[np.isfinite(err_mean)]).ravel(),
                np.abs(ideal_mean[np.isfinite(ideal_mean)]).ravel(),
            ]
        )
        amax = np.percentile(finite_abs, 99) if finite_abs.size else 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-amax, vmax=amax)

        mesh1 = axes[1].pcolormesh(
            ybins, xbins, err_mean, cmap="coolwarm", norm=norm, shading="auto"
        )
        axes[1].set_xlabel("Delta Angle (rad)")
        axes[1].set_ylabel("Delta Color (rad)")
        axes[1].set_title("Mean Signed Angle Error")
        fig.colorbar(mesh1, ax=axes[1], label="Radians")

        mesh2 = axes[2].pcolormesh(
            ybins, xbins, ideal_mean, cmap="coolwarm", norm=norm, shading="auto"
        )
        axes[2].set_xlabel("Delta Angle (rad)")
        axes[2].set_ylabel("Delta Color (rad)")
        axes[2].set_title("Mean Signed Ideal Error")
        fig.colorbar(mesh2, ax=axes[2], label="Radians")

        if "color_hat" in test_output:
            color_np = test_output["color"].numpy()
            color_hat_np = (test_output["color_hat"] % (2 * torch.pi)).numpy()

            axes[3].scatter(color_np, color_hat_np, alpha=0.5)
            axes[3].set_xlabel("True Color (rad)")
            axes[3].set_ylabel("Predicted Color (rad)")
            axes[3].set_title("Color Prediction")
            axes[3].axis("equal")

            err_c_np = test_output["error_color"].numpy()

            # Weighted sums and counts
            c_err_sum, xbins, ybins = np.histogram2d(
                delta_theta_np, delta_color_np, weights=err_c_np, bins=(40, 40)
            )

            c_counts, _, _ = np.histogram2d(
                delta_theta_np, delta_color_np, bins=(xbins, ybins)
            )

            # Mean signed error per bin (mask empty bins)
            c_err_mean = np.divide(
                c_err_sum,
                c_counts,
                out=np.full_like(c_err_sum, np.nan),
                where=c_counts > 0,
            )

            # Symmetric robust scale centered at 0
            c_finite_abs = np.abs(c_err_mean[np.isfinite(c_err_mean)]).ravel()

            c_amax = np.percentile(c_finite_abs, 99) if c_finite_abs.size else 1.0
            c_norm = TwoSlopeNorm(vcenter=0.0, vmin=-c_amax, vmax=c_amax)

            c_mesh1 = axes[4].pcolormesh(
                ybins, xbins, c_err_mean, cmap="coolwarm", norm=c_norm, shading="auto"
            )
            axes[4].set_xlabel("Delta Color (rad)")
            axes[4].set_ylabel("Delta Angle (rad)")
            axes[4].set_title("Mean Signed Color Error")
            fig.colorbar(mesh1, ax=axes[4], label="Radians")

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

    return


def calc_sliding_window_avg(delta_theta, error_theta):
    def circular_mean(angles: torch.Tensor) -> torch.Tensor:
        sin_sum = torch.sin(angles).mean(dim=0)
        cos_sum = torch.cos(angles).mean(dim=0)
        return torch.atan2(sin_sum, cos_sum)

    # Work in torch, then return NumPy
    sorted_idx = torch.argsort(delta_theta)
    delta_theta_sorted = delta_theta[sorted_idx]
    error_theta_sorted = error_theta[sorted_idx]

    window_width = np.pi / 10
    step_size = np.pi / 100
    x_vals = np.arange(-np.pi, np.pi, step_size)

    circ_mean_vals = []
    for x in x_vals:
        lower = x - window_width / 2
        upper = x + window_width / 2

        if lower < -np.pi:
            mask = (delta_theta_sorted >= (lower + 2 * np.pi)) | (
                delta_theta_sorted < upper
            )
        elif upper > np.pi:
            mask = (delta_theta_sorted >= lower) | (
                delta_theta_sorted < (upper - 2 * np.pi)
            )
        else:
            mask = (delta_theta_sorted >= lower) & (delta_theta_sorted < upper)

        values = error_theta_sorted[mask]
        if values.numel() == 0:
            circ_mean_vals.append(np.nan)
        else:
            circ_mean_vals.append(circular_mean(values).item())

    return x_vals, np.array(circ_mean_vals)
