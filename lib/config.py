import torch
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Callable
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


@dataclass
class ExperimentParams:
    # Core task setup
    dims: int = 1
    input_type: Literal["spatial", "feature"] = "spatial"
    input_size: int = 10  # spatial grid size per dimension
    p: float | List[float] = field(default_factory=lambda: [0.5])
    q: float = 0
    kappas: List[float] = field(default_factory=lambda: [8])
    task_type: Literal["binding", "tracking"] = "binding"

    # Model architecture
    hidden_size: int = 100
    output_type: Literal["angle", "feature", "angle_color", "feature_decode"] = "angle"

    # Stimulus tuning
    tuning_concentration: float = 3.0
    A: float = 1.0

    # RNN dynamics
    dt: float = 0.01
    tau: float = 0.25
    eps1: float = 0.8
    C: float = 0.01

    # Training
    lr: float = 0.001
    batch_size: int = 200
    num_batches: int = 2000
    interactive: bool = False

    # Testing
    test_batch_size: int = 0

    # Computed fields (set in __post_init__)
    eps2: float = field(init=False)
    device: torch.device = field(init=False)
    model: Optional[torch.nn.Module] = field(init=False)
    preprocess: Optional[Callable] = field(init=False)
    output_size: int = field(init=False)

    def __post_init__(self):
        # Derived parameter
        self.eps2 = (1.0 - self.eps1**2) ** 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Only set up the CNN if using feature inputs
        if self.input_type == "feature":
            weights = ConvNeXt_Base_Weights.DEFAULT
            model = convnext_base(weights=weights)
            model.to(self.device)
            model.eval()
            # Use only first two layers for feature extraction
            model.features = torch.nn.Sequential(*list(model.features.children())[:2])
            for p in model.features.parameters():
                p.requires_grad = False
            self.model = model
            self.preprocess = weights.transforms()
        else:
            self.model = None
            self.preprocess = None
        # output_size is set later after stimuli is constructed
