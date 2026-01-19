from abc import ABC, abstractmethod
import torch
from torch import pi
from torch.distributions import VonMises
import numpy as np
from scipy.special import iv as besselI
from .utils import generate_gabor_features


class DataGenerator(ABC):
    """
    Abstract Base Class for generating behavioral data and Ideal Observer estimates.
    """

    def __init__(
        self,
        dim: int,
        kappas: float | int | list | tuple,
        device: torch.device,
    ):
        if dim not in (1, 2):
            raise NotImplementedError(
                f"Can only generate 1D and 2D data, got {dim} dimensions"
            )

        self.dim = dim
        self.device = device

        # Normalize kappas to a list
        if isinstance(kappas, (float, int)):
            kappas = [float(kappas)]
        elif isinstance(kappas, (list, tuple)):
            kappas = list(kappas)

        if len(kappas) != dim and not (dim == 1 and len(kappas) == 1):
            # Note: Tracking is 1D spatial but generates 4 streams. dim=1 usually implies 1 kappa.
            raise ValueError(f"Expected {dim} kappa values, got {len(kappas)}")

        self.kappas = kappas
        # Create VonMises distributions for sampling noise on the device
        self.vms = [
            VonMises(torch.tensor(0.0, device=device), torch.tensor(k, device=device))
            for k in kappas
        ]

    # --- Shared Math Helpers (Numpy/CPU based for IO) ---

    def inv_logit(self, x):
        return 1 / (1 + np.exp(-x))

    def estimate_shared_log_likelihood(self, x1, x2, kappa):
        """Computes log likelihood of two points sharing a source (Von Mises)."""
        delta_x = np.abs(x1 - x2)
        delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
        kappa_eff = 2 * kappa * np.cos(delta_x / 2)

        log_num = np.log(besselI(0, kappa_eff))
        log_den = np.log((2 * np.pi) ** 2 * besselI(0, kappa) ** 2)
        return log_num - log_den

    def compute_circular_mean(self, angle1, angle2):
        x = np.cos(angle1) + np.cos(angle2)
        y = np.sin(angle1) + np.sin(angle2)
        return np.arctan2(y, x)

    def calc_mixture_circ_mean(self, theta, kappa_1, kappa_2, mu_1, mu_2):
        """
        Calculates the circular mean of a mixture of two Von Mises distributions.
        theta: Weight of component 1
        """
        r1 = besselI(1, kappa_1) / besselI(0, kappa_1)
        r2 = besselI(1, kappa_2) / besselI(0, kappa_2)
        z1 = r1 * np.exp(1j * mu_1)
        z2 = r2 * np.exp(1j * mu_2)
        z_mix = theta * z1 + (1 - theta) * z2
        return np.angle(z_mix)

    # --- Shared Torch Generators ---

    def _generate_angles(self, batch_size):
        return torch.rand(batch_size, device=self.device) * 2 * pi

    def _sample_noise(self, values, dim_idx):
        """Adds Von Mises noise to values."""
        return (values + self.vms[dim_idx].sample((len(values),))) % (2 * pi)

    # --- Abstract Methods ---

    @abstractmethod
    def _generate_batch(self, batch_size: int, test: bool, **kwargs) -> dict:
        """Generates the raw stimuli dictionary."""
        pass

    @abstractmethod
    def _ideal_observer(self, batch: dict) -> torch.Tensor:
        """Computes the Ideal Observer estimates for the batch."""
        pass

    def __call__(self, batch_size: int = 0, test=False, **kwargs):
        """
        Main entry point. Generates data -> Adds Noise -> Computes IO.
        """
        batch = self._generate_batch(batch_size, test, **kwargs)
        batch["ideal_observer_estimates"] = self._ideal_observer(batch)
        return batch


class BindingGenerator(DataGenerator):
    """
    Generator for the Feature Binding task (Target vs Distractor).
    Supports 1D (Angle) or 2D (Angle + Color).
    """

    def __init__(
        self,
        dim: int,
        kappas: float | list,
        p: float | list,
        q: float,
        device: torch.device,
    ):
        super().__init__(dim, kappas, device)

        # --- Binding-Specific Probability Setup ---
        if isinstance(p, (float, int)):
            ps = [float(p)] * dim
        elif isinstance(p, (list, tuple)):
            if len(p) != dim:
                raise ValueError(f"Expected {dim} p values, got {len(p)}")
            ps = [float(x) for x in p]
        else:
            raise TypeError("p must be a float/int or a sequence")

        if any((x < 0.0 or x > 1.0) for x in ps):
            raise ValueError(f"All p values must be in [0, 1], got {ps}")

        self.ps = ps
        self.q = float(q)

        # Validate Q (Correlation coeff)
        if self.dim == 2:
            p1, p2 = self.ps
            r = (p1 * (1.0 - p1) * p2 * (1.0 - p2)) ** 0.5
            if r > 0:
                q_min = -min(p1 * p2, (1.0 - p1) * (1.0 - p2)) / r
                q_max = min(p1 * (1.0 - p2), (1.0 - p1) * p2) / r
                if not (q_min <= q <= q_max):
                    raise ValueError(
                        f"Invalid q={q} for ps={ps}. Range: [{q_min:.4f}, {q_max:.4f}]"
                    )

    def _generate_batch(
        self, batch_size: int = 0, test: bool = False, **kwargs
    ) -> dict:
        structured_test = kwargs.get("structured_test", False)
        n_delta = kwargs.get("n_delta", 20)
        n_base = kwargs.get("n_base", 8)

        # --- Structured Test Grid Logic ---
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
                grid = torch.cartesian_prod(
                    base_angles, base_colors, delta_angles, delta_colors
                )
                target_angles = grid[:, 0]
                target_colors = grid[:, 1]
                distractor_angles = (target_angles + grid[:, 2]) % (2 * torch.pi)
                distractor_colors = (target_colors + grid[:, 3]) % (2 * torch.pi)
                targets = [target_angles, target_colors]
                distractors = [distractor_angles, distractor_colors]

        # --- Random Sampling Logic ---
        else:
            targets = [self._generate_angles(batch_size) for _ in range(self.dim)]
            distractors = [self._generate_angles(batch_size) for _ in range(self.dim)]

            if self.dim == 1:
                p1 = self.ps[0]
                if not test:
                    # Probabilistically replace distractor with target
                    mask = torch.rand(batch_size, device=self.device) < p1
                    distractors[0] = torch.where(mask, targets[0], distractors[0])
            else:
                p1, p2 = self.ps
                r = (p1 * (1.0 - p1) * p2 * (1.0 - p2)) ** 0.5
                prob = torch.tensor(
                    [
                        p1 * p2 + self.q * r,
                        p1 * (1.0 - p2) - self.q * r,
                        (1.0 - p1) * p2 - self.q * r,
                        (1.0 - p1) * (1.0 - p2) + self.q * r,
                    ],
                    device=self.device,
                )
                class_vector = prob.multinomial(batch_size, replacement=True)
                if not test:
                    distractors[0] = torch.where(
                        class_vector < 2, targets[0], distractors[0]
                    )
                    distractors[1] = torch.where(
                        (class_vector == 0) | (class_vector == 2),
                        targets[1],
                        distractors[1],
                    )

        # --- Add Noise ---
        if not test:
            targets_observed = [self._sample_noise(t, i) for i, t in enumerate(targets)]
            distractors_observed = [
                self._sample_noise(d, i) for i, d in enumerate(distractors)
            ]
        else:
            targets_observed = targets
            distractors_observed = distractors

        result = {}
        for i in range(self.dim):
            key = "angles" if i == 0 else "colors"
            result[f"target_{key}"] = targets[i]
            result[f"distractor_{key}"] = distractors[i]
            result[f"target_{key}_observed"] = targets_observed[i]
            result[f"distractor_{key}_observed"] = distractors_observed[i]
        return result

    def _ideal_observer(self, batch):
        if self.dim == 1:
            return self._compute_binding_io_1d(batch)
        else:
            return self._compute_binding_io_2d(batch)

    def _compute_binding_io_1d(self, batch):
        t_obs = batch["target_angles_observed"].detach().cpu().numpy()
        d_obs = batch["distractor_angles_observed"].detach().cpu().numpy()
        kappa = self.kappas[0]
        p = self.ps[0]

        # Calculate P(Shared)
        log_lik_shared = self.estimate_shared_log_likelihood(t_obs, d_obs, kappa)
        log_lik_indep = (1 / (2 * np.pi)) ** 2

        log_p1 = np.log(p) + log_lik_shared
        log_p2 = np.log(1 - p) + log_lik_indep
        p_shared = self.inv_logit(log_p1 - log_p2)

        # Calculate Mixture Parameters
        mu_shared = self.compute_circular_mean(t_obs, d_obs) % (2 * np.pi)

        delta_x = np.abs(t_obs - d_obs)
        delta_x = np.where(delta_x > np.pi, 2 * np.pi - delta_x, delta_x)
        kappa_eff = 2 * kappa * np.cos(delta_x / 2)

        estimates = []
        for i in range(len(t_obs)):
            est = self.calc_mixture_circ_mean(
                p_shared[i], kappa_eff[i], kappa, mu_shared[i], t_obs[i]
            )
            estimates.append(float(est % (2 * np.pi)))

        return torch.tensor(estimates, dtype=torch.float32, device=self.device)

    def _compute_binding_io_2d(self, batch):
        ta = batch["target_angles"].detach().cpu().numpy()
        tc = batch["target_colors"].detach().cpu().numpy()
        da = batch["distractor_angles"].detach().cpu().numpy()
        dc = batch["distractor_colors"].detach().cpu().numpy()

        angles_log_lik_shared = self.estimate_shared_log_likelihood(
            ta, da, self.kappas[0]
        )
        colors_log_lik_shared = self.estimate_shared_log_likelihood(
            tc, dc, self.kappas[1]
        )

        # Prior over 4 cases (same/same, same/diff, diff/same, diff/diff)
        p1, p2 = self.ps
        r = (p1 * (1.0 - p1) * p2 * (1.0 - p2)) ** 0.5
        log_prob = np.array(
            [
                np.log(p1 * p2 + self.q * r),
                np.log(p1 * (1.0 - p2) - self.q * r),
                np.log((1.0 - p1) * p2 - self.q * r),
                np.log((1.0 - p1) * (1.0 - p2) + self.q * r),
            ],
            dtype=np.float64,
        )

        log_2pi = np.log(2 * np.pi)

        # Per-sample log unnormalized posteriors over 4 cases
        c1 = log_prob[0] + angles_log_lik_shared + colors_log_lik_shared
        c2 = log_prob[1] + angles_log_lik_shared - 2 * log_2pi
        c3 = log_prob[2] + colors_log_lik_shared - 2 * log_2pi
        c4 = log_prob[3] - 4 * log_2pi.repeat(c1.size)

        logc = np.stack([c1, c2, c3, c4], axis=0)
        m = np.max(logc, axis=0, keepdims=True)
        post = np.exp(logc - m)
        post /= np.sum(post, axis=0, keepdims=True)

        p_shared_angle = post[0] + post[1]

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

        return torch.tensor(
            estimates,
            dtype=batch["target_angles"].dtype,
            device=batch["target_angles"].device,
        )


class TrackingGenerator(DataGenerator):
    """
    Generator for the Bayesian Tracking task (A, B -> C, D).
    Always 1D spatial tracking in this implementation.
    """

    def __init__(
        self,
        kappas: float,
        p_swap: float,
        device: torch.device,
    ):
        # Tracking is 1D, so dim=1
        super().__init__(dim=1, kappas=kappas, device=device)
        self.p_swap = float(p_swap)

    def _generate_batch(self, batch_size, test, **kwargs):
        # 1. Latents (x1, x2)
        x1 = self._generate_angles(batch_size)
        x2 = self._generate_angles(batch_size)

        # 2. Swap Decision
        swap = (torch.rand(batch_size, device=self.device) < self.p_swap).float()

        # 3. Measurements
        # Time 1: A ~ x1, B ~ x2
        # Time 2: C ~ (x1 or x2), D ~ (x2 or x1)
        c_mean = x1 * (1 - swap) + x2 * swap
        d_mean = x2 * (1 - swap) + x1 * swap

        # Add Noise (using dim_idx=0 since it's 1D tracking)
        if not test:
            a_obs = self._sample_noise(x1, 0)
            b_obs = self._sample_noise(x2, 0)
            c_obs = self._sample_noise(c_mean, 0)
            d_obs = self._sample_noise(d_mean, 0)
        else:
            a_obs, b_obs = x1, x2
            c_obs, d_obs = c_mean, d_mean

        return {
            "a": a_obs,
            "b": b_obs,
            "c": c_obs,
            "d": d_obs,
            "x1": x1,
            "x2": x2,
            "swap": swap,
        }

    def _ideal_observer(self, batch):
        # Extract numpy arrays
        a = batch["a"].detach().cpu().numpy()
        b = batch["b"].detach().cpu().numpy()
        c = batch["c"].detach().cpu().numpy()
        d = batch["d"].detach().cpu().numpy()

        kappa = self.kappas[0]

        # 1. Compute P(Stay)
        # Note: self.p_swap is P(Swap). p_prior usually refers to P(Stay).
        p_stay_prior = 1.0 - self.p_swap

        # Hypothesis 1: Stay (A->C, B->D)
        log_lik_stay = self.estimate_shared_log_likelihood(
            a, c, kappa
        ) + self.estimate_shared_log_likelihood(b, d, kappa)

        # Hypothesis 2: Swap (A->D, B->C)
        log_lik_swap = self.estimate_shared_log_likelihood(
            a, d, kappa
        ) + self.estimate_shared_log_likelihood(b, c, kappa)

        log_term_stay = np.log(p_stay_prior) + log_lik_stay
        log_term_swap = np.log(1.0 - p_stay_prior) + log_lik_swap

        # P(Stay | Data)
        p_stay_post = self.inv_logit(log_term_stay - log_term_swap)

        # 2. Candidate Means
        mu_stay = self.compute_circular_mean(a, c) % (2 * np.pi)
        mu_swap = self.compute_circular_mean(a, d) % (2 * np.pi)

        # 3. Effective Kappas (based on distance between pairs)
        def get_kappa_eff(x, y):
            dx = np.abs(x - y)
            dx = np.where(dx > np.pi, 2 * np.pi - dx, dx)
            return 2 * kappa * np.cos(dx / 2)

        k_stay = get_kappa_eff(a, c)
        k_swap = get_kappa_eff(a, d)

        # 4. Mixture Mean
        estimates = []
        for i in range(len(a)):
            est = self.calc_mixture_circ_mean(
                p_stay_post[i], k_stay[i], k_swap[i], mu_stay[i], mu_swap[i]
            )
            estimates.append(float(est % (2 * np.pi)))

        return torch.tensor(estimates, dtype=torch.float32, device=self.device)


class StimuliGenerator(ABC):
    """
    Abstract base for converting raw angle/color values into
    Neural Network inputs (Bumps, Gabors, etc).
    """

    def __init__(self, dim: int, device: torch.device):
        if dim not in (1, 2):
            raise NotImplementedError(
                f"Can only generate 1D and 2D data, got {dim} dim"
            )
        self.dim = dim
        self.device = device

    @abstractmethod
    def compute_representation(self, *features) -> torch.Tensor:
        pass

    def __call__(self, batch: dict, test: bool) -> tuple:
        """
        Consumes a batch dict from a DataGenerator.
        Detects if it's Binding or Tracking based on keys.
        """
        # --- CASE 1: TRACKING (a,b,c,d) ---
        if "a" in batch and "b" in batch:
            features_list = ["a", "b", "c", "d"]
            inputs = []
            for key in features_list:
                # Tracking is 1D (angles only), so single arg to compute_representation
                feat_val = batch[key]
                rep = self.compute_representation(feat_val).to(self.device)
                inputs.append(rep)
            # Returns (stim_a, stim_b, stim_c, stim_d)
            return tuple(inputs)

        # --- CASE 2: BINDING (target/distractor) ---
        else:
            suffix = "" if test else "_observed"
            feature_names = ["angles"] if self.dim == 1 else ["angles", "colors"]

            target_features = [
                batch[f"target_{name}{suffix}"] for name in feature_names
            ]
            distractor_features = [
                batch[f"distractor_{name}{suffix}"] for name in feature_names
            ]

            target_inputs = self.compute_representation(*target_features).to(
                self.device
            )
            distractor_inputs = self.compute_representation(*distractor_features).to(
                self.device
            )

            # Returns (target_input, distractor_input)
            return (target_inputs, distractor_inputs)


class SpatialStimuliGenerator(StimuliGenerator):
    """
    Generates Von Mises 'Bumps' (Population Coding).
    """

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
        if len(features) == 1:
            return self._bump_input_1d(features[0])
        elif len(features) == 2:
            return self._bump_input_2d(features[0], features[1])
        else:
            raise ValueError(f"Unexpected number of features: {len(features)}")

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
    """
    Generates Gabor patches passed through a CNN backbone.
    """

    def __init__(self, dim: int, model, device, preprocess):
        super().__init__(dim, device)
        self.model = model
        self.preprocess = preprocess

    @property
    def n_inputs(self) -> int:
        # Assumes model.features is defined
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
