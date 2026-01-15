import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.special import iv as besselI
from tqdm import tqdm
from torch.distributions import VonMises
import matplotlib.pyplot as plt


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


def my_loss(output, target):
    # Loss function that computes the mean squared error, with target being the angle in radians
    angle_x = torch.cos(target)
    angle_y = torch.sin(target)
    angle_x_hat = output[:, 0]
    angle_y_hat = output[:, 1]

    angle_x_errors = (angle_x_hat - angle_x) ** 2
    angle_y_errors = (angle_y_hat - angle_y) ** 2
    return (angle_x_errors + angle_y_errors).mean()


def inv_logit(x):
    return 1 / (1 + np.exp(-x))


def estimate_shared_log_likelihood(x1, x2, kappa):
    delta_x = np.abs(x1 - x2)
    delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
    kappa_eff = 2 * kappa * np.cos(delta_x / 2)
    log_num = np.log(besselI(0, kappa_eff))
    log_den = np.log((2 * np.pi) ** 2 * besselI(0, kappa) ** 2)
    return log_num - log_den


def compute_p_stay(x, kappa, p=0.5):
    log_lik_stay = estimate_shared_log_likelihood(
        x[0], x[2], kappa
    ) + estimate_shared_log_likelihood(x[1], x[3], kappa)
    log_lik_swap = estimate_shared_log_likelihood(
        x[0], x[3], kappa
    ) + estimate_shared_log_likelihood(x[1], x[2], kappa)

    log_p1 = np.log(p) + log_lik_stay
    log_p2 = np.log(1 - p) + log_lik_swap

    posterior_prob_shared = inv_logit(log_p1 - log_p2)
    return posterior_prob_shared


def calc_mixture_circ_mean(theta, kappa_1, kappa_2, mu_1, mu_2):
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


def compute_ideal_observer_estimates_1d(inputs: torch.Tensor, kappa_tilde, p_prior):
    """Compute ideal observer estimates for a batch of inputs"""
    estimates = []
    n = inputs.shape[0]

    inputs_np = inputs.detach().cpu().numpy()

    for i in range(n):
        p_shared = compute_p_stay(inputs_np[i, :], kappa_tilde, p_prior)
        mu_stay = compute_circular_mean(inputs_np[i, 0], inputs_np[i, 2]) % (2 * np.pi)
        mu_swap = compute_circular_mean(inputs_np[i, 0], inputs_np[i, 3]) % (2 * np.pi)

        delta_x_stay = abs(inputs_np[i, 0] - inputs_np[i, 2])
        delta_x_swap = abs(inputs_np[i, 0] - inputs_np[i, 3])
        delta_x_stay = (
            delta_x_stay if delta_x_stay < np.pi else 2 * np.pi - delta_x_stay
        )
        delta_x_swap = (
            delta_x_swap if delta_x_swap < np.pi else 2 * np.pi - delta_x_swap
        )

        kappa_stay = 2 * kappa_tilde * np.cos(delta_x_stay / 2)
        kappa_swap = 2 * kappa_tilde * np.cos(delta_x_swap / 2)

        est = calc_mixture_circ_mean(p_shared, kappa_stay, kappa_swap, mu_stay, mu_swap)
        estimates.append(float(est % (2 * np.pi)))

    return torch.tensor(estimates, dtype=inputs.dtype, device=inputs.device)


def get_circular_error(preds, targets):
    """Computes absolute angular error accounting for wrap-around."""
    diff = torch.abs(preds - targets)
    diff = torch.min(diff, 2 * torch.pi - diff)
    return diff


def generate_batch(batch_size, noise_dist, p):
    noise_dist = VonMises(torch.tensor(0), kappa)
    latents = 2 * torch.pi * torch.rand([batch_size, 2])
    noise = noise_dist.sample((batch_size, 4))
    swap = (torch.rand(batch_size) < p).float()

    x1, x2 = latents[:, 0], latents[:, 1]

    a = x1 + noise[:, 0]
    b = x2 + noise[:, 1]
    c = (x1 * (1 - swap) + x2 * swap) + noise[:, 2]
    d = (x2 * (1 - swap) + x1 * swap) + noise[:, 3]

    stimuli = torch.stack([a, b, c, d], dim=1)
    stimuli = (stimuli) % (2 * torch.pi)
    return stimuli, latents, swap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_model = MyModel(4, 100, 2)
my_model.to(device)

optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)

kappa = torch.tensor(7)
noise_dist = VonMises(torch.tensor(0), kappa)
p = 0.5

batch_size = 200
epochs = 4000
model_losses = []


for b in tqdm(range(epochs)):
    optimizer.zero_grad()
    stimuli, targets, _ = generate_batch(batch_size, noise_dist, p)
    stimuli = stimuli.to(device)
    targets = targets.to(device)
    output = my_model(stimuli)

    loss = my_loss(output, targets[:, 0])
    loss.backward()
    optimizer.step()
    model_losses.append(loss.item())
    if (b + 1) % 100 == 0:
        model_preds = torch.atan2(output[:, 1], output[:, 0]).detach()
        model_preds = model_preds % (2 * torch.pi)  # Normalize to [0, 2pi]

        io_preds = compute_ideal_observer_estimates_1d(stimuli, float(kappa), p)
        io_preds = io_preds % (2 * torch.pi)

        model_errors = (
            get_circular_error(model_preds, targets[:, 0]).detach().cpu().numpy()
        )
        io_errors = get_circular_error(io_preds, targets[:, 0]).detach().cpu().numpy()

        model_error = np.mean(model_errors - io_errors)
        tqdm.write(f"model error above ideal observer: {model_error}")
        plt.close("all")
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot 1: Loss progression
        ax.plot(model_losses, "b-", label="Model Loss", linewidth=2)
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig("feedforward_two_source_training.png", dpi=300, bbox_inches="tight")


def visualize_test_batch(model, kappa_val, p_val, batch_size=1000):
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device

    # 1. Generate Test Data
    # Note: We create a fresh noise distribution for testing to ensure independence
    test_dist = VonMises(torch.tensor(0.0), kappa_val)
    stimuli, targets, _ = generate_batch(batch_size, test_dist, p_val)
    stimuli = stimuli.to(device)
    targets = targets[:, 0].to(device)  # We only care about estimating x1

    with torch.no_grad():
        # 2. Get Model Predictions
        # Output is (cos, sin), convert to angle
        model_vec = model(stimuli)
        model_preds = torch.atan2(model_vec[:, 1], model_vec[:, 0])
        model_preds = model_preds % (2 * torch.pi)  # Normalize to [0, 2pi]

        # 3. Get Ideal Observer Predictions
        # Note: IO expects raw stimuli
        io_preds = compute_ideal_observer_estimates_1d(stimuli, float(kappa_val), p_val)
        io_preds = io_preds % (2 * torch.pi)

    # 4. Calculate Errors (vs Ground Truth)
    model_errors = get_circular_error(model_preds, targets).cpu().numpy()
    io_errors = get_circular_error(io_preds, targets).cpu().numpy()

    # Calculate deviation between Model and Ideal Observer
    deviation_from_ideal = get_circular_error(model_preds, io_preds).cpu().numpy()

    # --- PLOTTING ---
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Convert to CPU for plotting
    m_p = model_preds.cpu().numpy()
    i_p = io_preds.cpu().numpy()

    # PLOT 1: Model vs Ideal Observer (Scatter)
    # This checks: Did the network learn the Bayesian function?
    ax = axes[0]
    ax.scatter(i_p, m_p, alpha=0.1, s=10, c="black")
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "r--", lw=2, label="Perfect Match")
    ax.set_xlabel("Ideal Observer Estimate (rad)")
    ax.set_ylabel("Model Estimate (rad)")
    ax.set_title(
        f"Model vs. Ideal Observer\n(Mean Dev: {np.mean(deviation_from_ideal):.3f} rad)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PLOT 2: Error Histogram Comparison
    # This checks: Is the model as accurate as the IO?
    ax = axes[1]
    bins = np.linspace(0, np.pi / 2, 50)
    ax.hist(
        model_errors,
        bins=bins,
        alpha=0.5,
        label=f"Model (Mean: {np.mean(model_errors):.3f})",
        density=True,
        color="blue",
    )
    ax.hist(
        io_errors,
        bins=bins,
        alpha=0.5,
        label=f"Ideal Obs (Mean: {np.mean(io_errors):.3f})",
        density=True,
        color="orange",
    )
    ax.set_xlabel("Absolute Error vs Truth (rad)")
    ax.set_ylabel("Density")
    ax.set_title("Error Distribution Comparison")
    ax.legend()

    # PLOT 3: Deviation Histogram
    # This checks: How often does the model disagree with the math?
    ax = axes[2]
    ax.hist(deviation_from_ideal, bins=50, color="purple", alpha=0.7)
    ax.set_xlabel("Distance between Model and IO (rad)")
    ax.set_title("Model Deviation from Ideal Logic")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("test_batch_evaluation.png", dpi=300)
    print("Test complete. Visualization saved to 'test_batch_evaluation.png'.")
    print(
        f"Average deviation from Ideal Observer: {np.mean(deviation_from_ideal):.4f} radians"
    )


# ==========================================
# RUN THE VISUALIZATION
# ==========================================
# Run this after your training loop finishes
visualize_test_batch(my_model, kappa, p)
