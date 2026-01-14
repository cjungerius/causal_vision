import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Base_Weights, convnext_base
from lib.generators import DataGenerator, FeatureStimuliGenerator
from lib.utils import (
    circular_distance,
    visualize_test_output,
    decoder,
    generate_gabor_features,
)
from tqdm import tqdm
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


def my_loss_feature(target, outputs):
    errors = (outputs - target) ** 2
    return errors.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_g = DataGenerator(dim=2, kappas=[7, 7], p=[0.5, 0.5], q=0.5, device=device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ConvNeXt_Base_Weights.DEFAULT
model = convnext_base(weights=weights)
model.to(device)
model.eval()
model.features = torch.nn.Sequential(*list(model.features.children())[:2])
for p in model.features.parameters():
    p.requires_grad = False
    preprocess = weights.transforms()

f_g = FeatureStimuliGenerator(dim=2, model=model, preprocess=preprocess, device=device)
my_model = MyModel(256, 50, 128)

optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
batch_size = 10
epochs = 500
model_losses = []


for b in (tbar := tqdm(range(epochs))):
    optimizer.zero_grad()
    batch = d_g(batch_size)
    batch_stim = f_g(batch, test=False)
    batch_stim = torch.cat(batch_stim, dim=1)
    output = my_model(batch_stim)

    target = generate_gabor_features(
        batch["target_angles"],
        batch.get("target_colors", None),
        model,
        device,
        preprocess,
    )
    loss = my_loss_feature(target, output)
    loss.backward()
    optimizer.step()
    model_losses.append(loss.item())
    tbar.set_description(f"Progress (current loss: {loss.item()})")
    if (b + 1) % 100 == 0:
        tqdm.write(f"Batch {b}, Loss: {loss.item()}")
        plt.close("all")
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot 1: Loss progression
        ax.plot(model_losses, "b-", label="Model Loss", linewidth=2)
        ax.set_xlabel("Batch Number")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
#        plt.savefig("feedforward_training.png", dpi=300, bbox_inches="tight")

# add a test batch
test_batch = d_g(2000, test=True, structured_test=False)
test_batch_stim = torch.cat(f_g(test_batch, test=True), dim=1)
test_output = my_model(test_batch_stim)


def analyze_test_batch(batch, test_output):
    theta = batch["target_angles"].cpu()

    xy = decoder(features=test_output, use_nn_decoder=True, device="cpu", dims=2)
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

    test_output = {
        "theta": theta,
        "theta_hat": theta_hat,
        "delta_theta": delta_theta,
        "error_theta": error_theta,
        "ideal_observer": delta_ideal,
    }
    return test_output


analyzed_test = analyze_test_batch(test_batch, test_output)

visualize_test_output(analyzed_test, dims=1, fname="my_feedforward.png")
