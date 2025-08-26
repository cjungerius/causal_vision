# %%
from abc import ABC, abstractmethod
import torch
from torch import pi
from torch.distributions import VonMises
import numpy as np
from utils import generate_gabor_features
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Load the ConvNeXt model with pretrained weights
model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
# Set the model to evaluation mode
model.eval()
weights = ConvNeXt_Base_Weights.DEFAULT
preprocess = weights.transforms()
model.features = model.features[0:2]  # Use only the first two layers for feature extraction
for param in model.parameters():
    param.requires_grad = False
model.to(device)
# %%
class DataGenerator(ABC):
    def __init__(self, p: float, q: float, kappa_1: float, kappa_2: float):
        self.p = p
        self.q = q
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.vm_1 = VonMises(0, self.kappa_1)
        self.vm_2 = VonMises(0, self.kappa_2)

    def _generate_angles(self, batch_size):
        return torch.rand(batch_size) * 2 * pi

    @abstractmethod
    def generate_batch(self, batch_size: int) -> dict:
        pass

class DataGenerator1D(DataGenerator):
    def __init__(self, p, kappa):
        super().__init__(p, 0, kappa, 1)

    def generate_batch(self, batch_size: int):
        targets = self._generate_angles(batch_size)
        distractors = self._generate_angles(batch_size)
        distractors = torch.where(torch.rand(batch_size) < self.p, targets, distractors)
        
        targets_observed = (targets + self.vm_1.sample((batch_size,))) % (2 * pi)
        distractors_observed = (distractors + self.vm_1.sample((batch_size,))) % (2 * pi)

        return {
            "target_angles": targets,
            "distractor_angles": distractors,
            "target_angles_observed": targets_observed,
            "distractor_angles_observed": distractors_observed,
        }

class DataGenerator2D(DataGenerator):
    def __init__(self, p, q, kappa_1, kappa_2):
        super().__init__(p, q, kappa_1, kappa_2)

    def generate_batch(self, batch_size: int):
        target_angles = self._generate_angles(batch_size)
        target_colors = self._generate_angles(batch_size)
        distractor_angles = self._generate_angles(batch_size)
        distractor_colors = self._generate_angles(batch_size)

        probability_vector = torch.tensor([
            self.p * self.p + self.q * np.sqrt(self.p ** 2 * (1 - self.p) ** 2),
            self.p * (1 - self.p) - self.q * np.sqrt(self.p ** 2 * (1 - self.p) ** 2),
            (1 - self.p) * self.p - self.q * np.sqrt(self.p ** 2 * (1 - self.p) ** 2),
            (1 - self.p) ** 2 + self.q * np.sqrt(self.p ** 2 * (1 - self.p) ** 2)
        ])

        class_vector = probability_vector.multinomial(batch_size, replacement=True)
    
        distractor_angles = torch.where(torch.logical_or(class_vector == 0, class_vector == 1) , target_angles, distractor_angles)
        distractor_colors = torch.where(torch.logical_or(class_vector == 0, class_vector == 2), target_colors, distractor_colors)
        
        target_angles_observed = (target_angles + self.vm_1.sample((batch_size,))) % (2 * pi)
        target_colors_observed = (target_colors + self.vm_2.sample((batch_size,))) % (2 * pi)
        distractor_angles_observed = (distractor_angles + self.vm_1.sample((batch_size,))) % (2 * pi)
        distractor_colors_observed = (distractor_colors + self.vm_2.sample((batch_size,))) % (2 * pi)

        return {
            "target_angles": target_angles,
            "target_colors": target_colors,
            "distractor_angles": distractor_angles,
            "distractor_colors": distractor_colors,
            "target_angles_observed": target_angles_observed,
            "target_colors_observed": target_colors_observed,
            "distractor_angles_observed": distractor_angles_observed,
            "distractor_colors_observed": distractor_colors_observed,
        }

class StimuliGenerator(ABC):
    def __init__(self):
        return
    
    @abstractmethod
    def generate_inputs(self, batch: dict, test: bool) -> tuple:
        pass

class SpatialStimuliGenerator1D(StimuliGenerator):
    def __init__(self, n_a, tuning_concentration, A):
        super().__init__()
        self.n_a = n_a
        self.tuning_concentration = tuning_concentration
        self.A = A

    def bump_input(self, angles):
        batch_size = angles.shape[0]
        device = angles.device
        # preferred directions: 0, 2π/n, ..., 2π*(n-1)/n
        unit_indices = torch.arange(self.n_a, device=device, dtype=angles.dtype)  # [0..n-1]
        preferred = (2 * pi) * unit_indices / self.n_a  # [n]
        angle_diffs = angles.unsqueeze(1) - preferred.unsqueeze(0)  # [B, n]
        bumps = self.A * torch.exp(self.tuning_concentration * torch.cos(angle_diffs))
        return bumps

    def generate_inputs(self, batch, test=False):
        if test:
            target_inputs = self.bump_input(batch["target_angles"])
            distractor_inputs = self.bump_input(batch["distractor_angles"])
        else:
            target_inputs = self.bump_input(batch["target_angles_observed"])
            distractor_inputs = self.bump_input(
                batch["distractor_angles_observed"])
        return (target_inputs, distractor_inputs)


class SpatialStimuliGenerator2D(StimuliGenerator):
    def __init__(self, n_a, tuning_concentration, A):        
        super().__init__()
        self.n_a = n_a
        self.tuning_concentration = tuning_concentration
        self.A = A

    def bump_input(self, angles, colors):
        batch_size = angles.shape[0]
        device = angles.device
        idx = torch.arange(self.n_a, device=device, dtype=angles.dtype)  # [0..n-1]
        pref = (2 * pi) * idx / self.n_a  # [n]

        diffs_x = angles.unsqueeze(1) - pref.unsqueeze(0)  # [B, n]
        diffs_y = colors.unsqueeze(1) - pref.unsqueeze(0)  # [B, n]

        bumps_x = self.A * torch.exp(self.tuning_concentration * torch.cos(diffs_x))  # [B, n]
        bumps_y = self.A * torch.exp(self.tuning_concentration * torch.cos(diffs_y))  # [B, n]

        # Outer product per batch -> [B, n, n]
        bumps = torch.einsum('bi,bj->bij', bumps_x, bumps_y)
        norm = torch.exp(torch.tensor(self.tuning_concentration, device=device))
        bumps = bumps / norm
        return bumps.view(batch_size, -1)
    
    def generate_inputs(self, batch, test=False):
        if test:
            target_inputs = self.bump_input(batch["target_angles"], batch["target_colors"])
            distractor_inputs = self.bump_input(batch["distractor_angles"], batch["distractor_colors"])
        else:
            target_inputs = self.bump_input(batch["target_angles_observed"], batch["target_colors_observed"])
            distractor_inputs = self.bump_input(
                batch["distractor_angles_observed"],
                batch["distractor_colors_observed"]
                )
            
        return (target_inputs, distractor_inputs)

class FeatureStimuliGenerator1D(StimuliGenerator):
    def __init__(self, model, device, preprocess):
        super().__init__()
        self.model = model
        self.device = device
        self.preprocess = preprocess
    
    def generate_inputs(self, batch, test=False):
        colors = torch.zeros(batch["target_angles"].shape[0])

        if test:
            target_inputs = generate_gabor_features(batch["target_angles"], colors, self.model, self.device, self.preprocess)
            distractor_inputs = generate_gabor_features(batch["distractor_angles"], colors, self.model, self.device, self.preprocess)
        else:
            target_inputs = generate_gabor_features(batch["target_angles_observed"], colors, self.model, self.device, self.preprocess)
            distractor_inputs = generate_gabor_features(batch["distractor_angles_observed"], colors, self.model, self.device, self.preprocess)
        return (target_inputs, distractor_inputs)


class FeatureStimuliGenerator2D(StimuliGenerator):
    def __init__(self, model, device, preprocess):
        super().__init__()
        self.model = model
        self.device = device
        self.preprocess = preprocess
    
    def generate_inputs(self, batch, test=False):

        if test:
            target_inputs = generate_gabor_features(batch["target_angles"], batch["target_colors"], self.model, self.device, self.preprocess)
            distractor_inputs = generate_gabor_features(batch["distractor_angles"], batch["distractor_colors"], self.model, self.device, self.preprocess)
        else:
            target_inputs = generate_gabor_features(batch["target_angles_observed"], batch["target_colors_observed"], self.model, self.device, self.preprocess)
            distractor_inputs = generate_gabor_features(batch["distractor_angles_observed"], batch["distractor_colors_observed"], self.model, self.device, self.preprocess)
        return (target_inputs, distractor_inputs)
        
# %%
datgen1d = DataGenerator1D(0.5, 8)
stimgen1d = SpatialStimuliGenerator1D(10, 3, 1)
b = datgen1d.generate_batch(20)
target_inputs, distractor_inputs = stimgen1d.generate_inputs(b)
from matplotlib import pyplot as plt
plt.plot(target_inputs[7,:])

# %%
datgen2d = DataGenerator2D(0.5,0.5,8,8)
stimgen2d = SpatialStimuliGenerator2D(10, 3, 1)
b = datgen2d.generate_batch(20)
target_inputs, distractor_inputs = stimgen2d.generate_inputs(b)
# %%
plt.plot(target_inputs[1,:])

# %%
