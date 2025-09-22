# %%
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt

import torch
from torch import pi
from torch.distributions import VonMises
from torchvision.models import ConvNeXt_Base_Weights, convnext_base
from .utils import generate_gabor_features

import numpy as np
from scipy.special import iv as besselI


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

    def inv_logit(self, x):
        return 1 / (1 + np.exp(-x))

    def estimate_shared_likelihood(self, x1, x2, kappa):
        delta_x = np.abs(x1 - x2)
        delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
        kappa_eff = 2 * kappa * np.cos(delta_x / 2)
        marginal_likelihood = besselI(0, kappa_eff) / (
            (2 * np.pi) ** 2 * besselI(0, kappa) ** 2
        )
        return marginal_likelihood

    def compute_p_shared(self, x1, x2, kappa, p=0.5):
        lik_shared = self.estimate_shared_likelihood(x1, x2, kappa)
        lik_indep = (1 / (2 * np.pi)) ** 2

        log_p1 = np.log(p) + np.log(lik_shared)
        log_p2 = np.log(1 - p) + np.log(lik_indep)

        posterior_prob_shared = self.inv_logit(log_p1 - log_p2)
        return posterior_prob_shared

    def calc_mixture_circ_mean(self, theta, kappa_1, kappa_2, mu_1, mu_2):
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

    def compute_circular_mean(self, angle1, angle2):
        """Compute circular mean of two angles"""
        # Convert to unit vectors and average
        x = np.cos(angle1) + np.cos(angle2)
        y = np.sin(angle1) + np.sin(angle2)
        return np.arctan2(y, x)

    def compute_ideal_observer_estimates_1d(
        self, first_inputs, second_inputs, kappa_tilde, p_prior
    ):
        """Compute ideal observer estimates for a batch of inputs"""
        estimates = []

        first_np = first_inputs.detach().cpu().numpy()
        second_np = second_inputs.detach().cpu().numpy()

        for i in range(len(first_np)):
            p_shared = self.compute_p_shared(
                first_np[i], second_np[i], kappa_tilde, p_prior
            )
            mu_shared = self.compute_circular_mean(first_np[i], second_np[i]) % (2 * np.pi)

            delta_x = abs(first_np[i] - second_np[i])
            if delta_x > np.pi:
                delta_x = 2 * np.pi - delta_x
            kappa_eff = 2 * kappa_tilde * np.cos(delta_x / 2)

            est = self.calc_mixture_circ_mean(
                p_shared, kappa_eff, kappa_tilde, mu_shared, first_np[i]
            )
            estimates.append(float(est % (2 * np.pi)))

        return torch.tensor(estimates, dtype=first_inputs.dtype, device=first_inputs.device)

    def compute_ideal_observer_estimates_2d(
        self, target_angles, target_colors, distractor_angles, distractor_colors
    ):
        # Convert to numpy for likelihood calcs and stable normalization
        ta = target_angles.detach().cpu().numpy()
        tc = target_colors.detach().cpu().numpy()
        da = distractor_angles.detach().cpu().numpy()
        dc = distractor_colors.detach().cpu().numpy()

        angles_lik_shared = self.estimate_shared_likelihood(ta, da, self.kappas[0])
        colors_lik_shared = self.estimate_shared_likelihood(tc, dc, self.kappas[1])

        # Prior over 4 cases (same/same, same/diff, diff/same, diff/diff)
        p1, p2 = self.ps
        r = (p1 * (1.0 - p1) * p2 * (1.0 - p2)) ** 0.5
        prob = np.array(
            [
                p1 * p2 + self.q * r,
                p1 * (1.0 - p2) - self.q * r,
                (1.0 - p1) * p2 - self.q * r,
                (1.0 - p1) * (1.0 - p2) + self.q * r,
            ],
            dtype=np.float64,
        )

        log_2pi = np.log(2 * np.pi)

        # Per-sample log unnormalized posteriors over 4 cases
        c1 = np.log(prob[0]) + np.log(angles_lik_shared) + np.log(colors_lik_shared)
        c2 = np.log(prob[1]) + np.log(angles_lik_shared) - 2 * log_2pi
        c3 = np.log(prob[2]) + np.log(colors_lik_shared) - 2 * log_2pi
        c4 = np.log(prob[3]) - 4 * log_2pi.repeat(c1.size)


        logc = np.stack([c1, c2, c3, c4], axis=0)              # (4, B)
        m = np.max(logc, axis=0, keepdims=True)                # stabilize per-sample
        post = np.exp(logc - m)
        post /= np.sum(post, axis=0, keepdims=True)            # normalize per-sample

        # Posterior that the angle dimension is "shared"
        p_shared_angle = post[0] + post[1]                     # (B,)

        # Shared-component params for angle
        delta_x = np.abs(ta - da)
        delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
        kappa_eff = 2 * self.kappas[0] * np.cos(delta_x / 2)
        mu_shared = self.compute_circular_mean(ta, da) % (2 * np.pi)

        estimates = []
        for i in range(ta.shape[0]):
            est = self.calc_mixture_circ_mean(
                p_shared_angle[i], kappa_eff[i], self.kappas[0], mu_shared[i], ta[i]
            )
            estimates.append(float(est % (2 * np.pi)))

        return torch.tensor(estimates, dtype=target_angles.dtype, device=target_angles.device)

    def _ideal_observer(self, batch):
        if self.dim == 1:
            estimates = self.compute_ideal_observer_estimates_1d(
                batch["target_angles_observed"],
                batch["distractor_angles_observed"],
                self.kappas[0],
                self.ps[0],
            )
        else:
            estimates = self.compute_ideal_observer_estimates_2d(
                batch["target_angles_observed"],
                batch["target_colors_observed"],
                batch["distractor_angles_observed"],
                batch["distractor_colors_observed"],
            )
        return estimates

    def __call__(self, batch_size: int = None, test=False, structured_test=False, n_delta=20, n_base=8):
        """
        Generates a batch of data. If structured_test is True, generates a grid of all combinations
        of delta and base angles/colors. Otherwise, samples randomly.
        Returns a dict with the same keys regardless of mode.
        """
        # --- Generate target/distractor values ---
        if structured_test:
            if self.dim == 1:
                deltas = torch.linspace(-torch.pi, torch.pi, n_delta)
                bases = torch.linspace(0, 2 * torch.pi, n_base)
                grid = torch.cartesian_prod(bases, deltas)
                target_angles = grid[:, 0]
                distractor_angles = (target_angles + grid[:, 1]) % (2 * torch.pi)
                targets = [target_angles]
                distractors = [distractor_angles]

            else:
                delta_angles = torch.linspace(-torch.pi, torch.pi, n_delta)
                delta_colors = torch.linspace(-torch.pi, torch.pi, n_delta)
                base_angles = torch.linspace(0, 2 * torch.pi, n_base)
                base_colors = torch.linspace(0, 2 * torch.pi, n_base)
                grid = torch.cartesian_prod(base_angles, base_colors, delta_angles, delta_colors)
                target_angles = grid[:, 0]
                target_colors = grid[:, 1]
                distractor_angles = (target_angles + grid[:, 2]) % (2 * torch.pi)
                distractor_colors = (target_colors + grid[:, 3]) % (2 * torch.pi)
                targets = [target_angles, target_colors]
                distractors = [distractor_angles, distractor_colors]

        else:
            targets = [self._generate_angles(batch_size) for _ in range(self.dim)]
            distractors = [self._generate_angles(batch_size) for _ in range(self.dim)]

            if self.dim == 1:
                p1 = self.ps[0]
                if not test:
                    distractors[0] = torch.where(
                        torch.rand(batch_size) < p1, targets[0], distractors[0]
                    )
            else:
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
                if not test:
                    distractors[0] = torch.where(class_vector < 2, targets[0], distractors[0])
                    distractors[1] = torch.where(
                        (class_vector == 0) | (class_vector == 2), targets[1], distractors[1]
                    )

        # --- Add noise (observed) ---
        targets_observed = [self._sample_noise(t, i) for i, t in enumerate(targets)]
        distractors_observed = [self._sample_noise(d, i) for i, d in enumerate(distractors)]

        # --- Prepare output dictionary ---
        result = {}
        for i in range(self.dim):
            key = "angles" if i == 0 else "colors"
            result[f"target_{key}"] = targets[i].to(self.device)
            result[f"distractor_{key}"] = distractors[i].to(self.device)
            result[f"target_{key}_observed"] = targets_observed[i].to(self.device)
            result[f"distractor_{key}_observed"] = distractors_observed[i].to(self.device)

        # --- Ideal observer ---
        result["ideal_observer_estimates"] = self._ideal_observer(result).to(self.device)
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

        target_inputs = self.compute_representation(*target_features).to(self.device)
        distractor_inputs = self.compute_representation(*distractor_features).to(self.device)

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
            return generate_gabor_features(
                features[0], None, self.model, self.device, self.preprocess
            )
        else:
            return generate_gabor_features(
                features[0], features[1], self.model, self.device, self.preprocess
            )



