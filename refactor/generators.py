# %%
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

import torch
from torch import pi
from torch.distributions import VonMises
from torchvision.models import ConvNeXt_Base_Weights, convnext_base
from .utils import generate_gabor_features

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Load the ConvNeXt model with pretrained weights
model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
model.to(device)
# Set the model to evaluation mode
model.eval()
weights = ConvNeXt_Base_Weights.DEFAULT
preprocess = weights.transforms()
model.features = model.features[
    0:2
]  # Use only the first two layers for feature extraction
for param in model.parameters():
    param.requires_grad = False


# %%
class DataGenerator:
    def __init__(
        self,
        dim: int,
        kappas: float | int | list | tuple,
        p: float | int | list | tuple,
        q: float,
        device: torch.device,
    ):
        if dim not in (1, 2):
            raise NotImplementedError(
                f"Can only generate 1D and 2D data, got {dim} dimensions"
            )

        self.dim = dim

        # Normalize p to per-dimension list
        if isinstance(p, (float, int)):
            ps = [float(p)] * dim
        elif isinstance(p, (list, tuple)):
            if len(p) != dim:
                raise ValueError(f"Expected {dim} p values, got {len(p)}")
            ps = [float(x) for x in p]
        else:
            raise TypeError("p must be a float/int or a sequence of length dim")

        if any((x < 0.0 or x > 1.0) for x in ps):
            raise ValueError(f"All p values must be in [0, 1], got {ps}")

        self.ps = ps
        self.p = self.ps[0]  # backward-compat

        # Compute admissible q-range so all four joint probabilities are non-negative
        if self.dim == 1:
            q_min = 0.0
            q_max = 0.0
        else:
            p1, p2 = self.ps
            r = (p1 * (1.0 - p1) * p2 * (1.0 - p2)) ** 0.5
            if r == 0.0:
                q_min = 0.0
                q_max = 0.0
            else:
                q_min = -min(p1 * p2, (1.0 - p1) * (1.0 - p2)) / r
                q_max = min(p1 * (1.0 - p2), (1.0 - p1) * p2) / r

        if not (q_min <= q <= q_max):
            raise ValueError(
                f"For p={self.ps}, valid q must lie between {q_min:.6f} and {q_max:.6f}, got {q}"
            )
        self.q = float(q)

        self.device = device

        if isinstance(kappas, (float, int)):
            kappas = [float(kappas)]
        elif isinstance(kappas, (list, tuple)):
            kappas = list(kappas)

        if len(kappas) != dim:
            raise ValueError(f"Expected {dim} kappa values, got {len(kappas)}")

        self.kappas = kappas
        self.vms = [VonMises(0, k) for k in kappas]

    def _generate_angles(self, batch_size):
        return torch.rand(batch_size) * 2 * pi

    def _sample_noise(self, values, idx):
        return (values + self.vms[idx].sample((len(values),))) % (2 * pi)

    def __call__(self, batch_size: int):
        targets = [self._generate_angles(batch_size) for _ in range(self.dim)]
        distractors = [self._generate_angles(batch_size) for _ in range(self.dim)]

        if self.dim == 1:
            # For 1D: Replace distractors with targets with prob p1
            p1 = self.ps[0]
            distractors[0] = torch.where(
                torch.rand(batch_size) < p1, targets[0], distractors[0]
            )
        else:
            # For 2D with different marginals p1, p2 and coupling q
            p1, p2 = self.ps
            r = (p1 * (1.0 - p1) * p2 * (1.0 - p2)) ** 0.5
            prob = torch.tensor(
                [
                    p1 * p2 + self.q * r,  # (same, same)
                    p1 * (1.0 - p2) - self.q * r,  # (same, diff)
                    (1.0 - p1) * p2 - self.q * r,  # (diff, same)
                    (1.0 - p1) * (1.0 - p2) + self.q * r,  # (diff, diff)
                ]
            )
            class_vector = prob.multinomial(batch_size, replacement=True)

            # Replace distractors based on combination index
            distractors[0] = torch.where(class_vector < 2, targets[0], distractors[0])
            distractors[1] = torch.where(
                (class_vector == 0) | (class_vector == 2), targets[1], distractors[1]
            )

        # Add noise
        targets_observed = [self._sample_noise(t, i) for i, t in enumerate(targets)]
        distractors_observed = [
            self._sample_noise(d, i) for i, d in enumerate(distractors)
        ]

        # Prepare output dictionary
        result = {}
        for i in range(self.dim):
            key = "angles" if i == 0 else "colors"
            result[f"target_{key}"] = targets[i].to(self.device)
            result[f"distractor_{key}"] = distractors[i].to(self.device)
            result[f"target_{key}_observed"] = targets_observed[i].to(self.device)
            result[f"distractor_{key}_observed"] = distractors_observed[i].to(
                self.device
            )

        return result


class StimuliGenerator(ABC):
    def __init__(self, dim: int, device: torch.device):
        if dim not in (1, 2):
            raise NotImplementedError(
                f"Can only generate 1D and 2D data, got {dim} dimensions"
            )
        self.dim = dim
        self.device = device

    @abstractmethod
    def compute_representation(self, *features) -> torch.Tensor:
        pass

    def __call__(self, batch: dict, test: bool) -> tuple:
        suffix = "" if test else "_observed"
        # Extract features based on dim
        feature_names = ["angles"] if self.dim == 1 else ["angles", "colors"]
        target_features = [batch[f"target_{name}{suffix}"] for name in feature_names]
        distractor_features = [
            batch[f"distractor_{name}{suffix}"] for name in feature_names
        ]

        target_inputs = self.compute_representation(*target_features).to(device)
        distractor_inputs = self.compute_representation(*distractor_features).to(device)

        return (target_inputs, distractor_inputs)


class SpatialStimuliGenerator(StimuliGenerator):
    def __init__(
        self,
        dim: int,
        n_a: int,
        tuning_concentration: float,
        A: float,
        device: torch.device,
    ):
        super().__init__(dim, device)
        self.n_a = n_a
        self.tuning_concentration = tuning_concentration
        self.A = A

    @property
    def n_inputs(self) -> int:
        return self.n_a if self.dim == 1 else self.n_a * self.n_a

    def compute_representation(self, *features):
        if self.dim == 1:
            angles = features[0]
            return self._bump_input_1d(angles)
        else:
            return self._bump_input_2d(features[0], features[1])

    def _bump_input_1d(self, angles):
        device = angles.device
        idx = torch.arange(self.n_a, device=device, dtype=angles.dtype)
        preferred = (2 * pi) * idx / self.n_a
        diffs = angles.unsqueeze(1) - preferred.unsqueeze(0)
        bumps = self.A * torch.exp(self.tuning_concentration * torch.cos(diffs))
        return bumps

    def _bump_input_2d(self, angles, colors):
        device = angles.device
        idx = torch.arange(self.n_a, device=device, dtype=angles.dtype)
        preferred = (2 * pi) * idx / self.n_a

        diffs_x = angles.unsqueeze(1) - preferred.unsqueeze(0)
        diffs_y = colors.unsqueeze(1) - preferred.unsqueeze(0)

        bumps_x = self.A * torch.exp(self.tuning_concentration * torch.cos(diffs_x))
        bumps_y = self.A * torch.exp(self.tuning_concentration * torch.cos(diffs_y))

        bumps = torch.einsum("bi,bj->bij", bumps_x, bumps_y)
        norm = torch.exp(torch.tensor(self.tuning_concentration, device=device))
        return (bumps / norm).view(angles.shape[0], -1)


class FeatureStimuliGenerator(StimuliGenerator):
    def __init__(self, dim: int, model, device, preprocess):
        super().__init__(dim, device)
        self.model = model
        self.preprocess = preprocess

    @property
    def n_inputs(self) -> int:
        return [p.size() for p in self.model.features.parameters()][-1][0]

    def compute_representation(self, *features):
        if self.dim == 1:
            colors = torch.zeros_like(features[0])  # fake color for 1D
            return generate_gabor_features(
                features[0], colors, self.model, self.device, self.preprocess
            )
        else:
            return generate_gabor_features(
                features[0], features[1], self.model, self.device, self.preprocess
            )


# %%
# %%
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==== 1D ring demo (spatial stimuli) ====
    datgen1d = DataGenerator(dim=1, kappas=8.0, p=0.5, q=0, device=device)
    stimgen1d = SpatialStimuliGenerator(
        dim=1, n_a=64, tuning_concentration=3.0, A=1.0, device=device
    )

    batch1d = datgen1d(batch_size=64)
    target_1d_train, distractor_1d_train = stimgen1d(
        batch1d, test=False
    )  # noisy/observed
    target_1d_test, _ = stimgen1d(batch1d, test=True)  # clean

    plt.figure(figsize=(6, 3))
    plt.title("1D target (train) vs. clean (test)")
    plt.plot(target_1d_train[0].cpu().numpy(), label="train/observed")
    plt.plot(target_1d_test[0].cpu().numpy(), label="test/clean", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==== 2D grid demo (spatial stimuli) ====
    n_a_2d = 16  # grid will be n_a x n_a when flattened
    datgen2d = DataGenerator(
        dim=2, kappas=(8.0, 1.5), p=[0.5, 0.2], q=0.2, device=device
    )
    stimgen2d = SpatialStimuliGenerator(
        dim=2, n_a=n_a_2d, tuning_concentration=3.0, A=1.0, device=device
    )

    batch2d = datgen2d(batch_size=32)
    target_2d_train, _ = stimgen2d(batch2d, test=False)  # shape: [B, n_a*n_a]

    plt.figure(figsize=(4, 4))
    plt.title("2D target (train), sample 0")
    plt.imshow(target_2d_train[0].reshape(n_a_2d, n_a_2d).cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # ==== Feature-based demo (ConvNeXt+gabors) ====
    # Uses your existing model/preprocess defined above.
    featgen2d = FeatureStimuliGenerator(
        dim=2, model=model, device=device, preprocess=preprocess
    )

    small_batch = datgen2d(batch_size=4)
    with torch.no_grad():
        feat_target, feat_distr = featgen2d(small_batch, test=False)

    print(
        "Feature stimuli shapes (target, distractor):",
        tuple(feat_target.shape),
        tuple(feat_distr.shape),
    )
# %%
