import torch
from lib.generators import (
    DataGenerator,
    SpatialStimuliGenerator,
    FeatureStimuliGenerator,
)
from lib.rnn import RNN
from lib.trial import Trial
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from lib.utils import criterion, my_loss_spatial, analyze_test_batch, vizualize_test_output
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Callable

import matplotlib.pyplot as plt


@dataclass
class ExperimentParams:
    # Core task setup
    dims: int = 1
    input_type: Literal["spatial", "feature"] = "spatial"
    input_size: int = 10  # spatial grid size per dimension
    p: float | List[float] = field(default_factory=lambda: [0.5])
    q: float = 0
    kappas: List[float] = field(default_factory=lambda: [8])

    # Model architecture
    hidden_size: int = 100
    output_type: Literal["angle", "feature", "angle_color"] = "angle"

    # Stimulus tuning
    tuning_concentration: float = 8.0
    A: float = 1.0

    # RNN dynamics
    dt: float = 0.01
    tau: float = 0.25
    eps1: float = 0.8
    C: float = 0.1

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

def run_experiment(
    dims: int = 1,
    input_type: Literal["spatial", "feature"] = "spatial",
    input_size: int = 10,
    p: List[float] = [0.5],
    q: float = 0,
    kappas: List[float] = [8],
    hidden_size: int = 100,
    output_type: Literal["angle", "feature", "angle_color"] = "angle",
    tuning_concentration: float = 8.0,
    A: float = 1.0,
    dt: float = 0.01,
    tau: float = 0.25,
    eps1: float = 0.8,
    C: float = 0.01,
    lr: float = 0.001,
    batch_size: int = 200,
    num_batches: int = 2000,
    interactive: bool = False,
    test_batch_size: int = 0
):
    # a much simpler refactor, based on much simpler ideas.
    # step 0: set *all* the parameters we need everywhere in the trial (maybe we modularize these later)
    params = ExperimentParams(
        dims=dims,
        input_type=input_type,
        input_size=input_size,
        p=p,
        q=q,
        kappas=kappas,
        hidden_size=hidden_size,
        output_type=output_type,
        tuning_concentration=tuning_concentration,
        A=A,
        dt=dt,
        tau=tau,
        eps1=eps1,
        C=C,
        lr=lr,
        batch_size=batch_size,
        num_batches=num_batches,
        interactive=interactive,
        test_batch_size=test_batch_size,
    )

    # first: what is the task? 1 dimension, 2 dimensions, which input type? use some logic to make the correct data and stimuli generators.
    datagen = DataGenerator(
        params.dims, params.kappas, params.p, params.q, params.device
    )
    if params.input_type == "spatial":
        stimuli = SpatialStimuliGenerator(
            params.dims,
            params.input_size,
            params.tuning_concentration,
            params.A,
            params.device,
        )
    elif params.input_type == "feature":
        stimuli = FeatureStimuliGenerator(
            params.dims, params.model, params.device, params.preprocess
        )
    else:
        raise ValueError(f"invalid input type {params.input_type} specified")

    # Derive sizes after stimuli is known
    rnn_input_size = stimuli.n_inputs
    if params.output_type == "angle":
        params.output_size = 2
    elif params.output_type == "angle_color":
        params.output_size = 4
    else:  # "feature"
        params.output_size = rnn_input_size

    # next: how does a trial look, i.e., what are the phases the network goes through, when does it get which inputs, and when does it output?
    trial = Trial(params.dt, params.C, params.eps1, datagen, stimuli)
    # T_prestim = 0.1     # in seconds
    trial.add_phase("prestim", 0.1, "blank", False)
    # T_stim_1 = 0.25
    trial.add_phase("stim_1", 0.25, "distractor", False)
    # T_stim_2 = 0.25
    trial.add_phase("stim_2", 0.25, "target", False)
    # T_delay = 0.0
    # T_resp = 0.1
    trial.add_phase("resp", 0.1, "blank", True)

    # next up: what does the model look like? input depends on the stimuli generator, hidden state is defined by us, output depends on the output type
     
    rnn = RNN(
        rnn_input_size,
        params.hidden_size,
        params.output_size,
        params.dt,
        params.tau,
    )
    rnn.to(params.device)

    # define the training machinery: optimizer and so forth.
    opt = torch.optim.Adam(rnn.parameters(), params.lr)
    # now at long last, we can run the training!
    losses = []
    epochs = (
        tqdm(range(params.num_batches))
        if params.interactive
        else range(params.num_batches)
    )
    for b in epochs:
        opt.zero_grad()
        batch = trial.run_batch(rnn, params.batch_size, False)
        batch_loss = criterion(batch, params)
        batch_loss.backward()
        opt.step()
        losses.append(batch_loss.item())

        if (b + 1) % 50 == 0:
            print(f"\nbatch {b + 1} loss: {batch_loss}")
            if params.output_type == "angle":
                with torch.no_grad():
                    io_angles = batch["ideal_observer_estimates"].to(
                        params.device
                    )  # [B]
                    io_vec = torch.stack(
                        [torch.cos(io_angles), torch.sin(io_angles)], dim=-1
                    ).unsqueeze(1)  # [B,1,2]
                    ideal_loss = my_loss_spatial(batch["target_angles"], io_vec)
                print(f"ideal observer loss: {ideal_loss.item()}")
    
    # After training, we can get a test batch to see how our model behaves

    if params.test_batch_size > 0:

        with torch.no_grad():
            test_batch = trial.run_batch(rnn, params.test_batch_size, True)

    experiment_output = {"params": params, "rnn": rnn, "losses": losses, "test_batch": test_batch}

    return experiment_output

def add_test_batch(experiment_output, test_batch_size):
    # In case you ran an experiment without a test batch and want to generate one later
    test_data_gen = DataGenerator(
        experiment_output["params"].dims,
        experiment_output["params"].kappas,
        experiment_output["params"].p,
        experiment_output["params"].q,
        experiment_output["params"].device
    )

    if experiment_output["params"].input_type == "spatial":
        stimuli = SpatialStimuliGenerator(
            experiment_output["params"].dims,
            experiment_output["params"].input_size,
            experiment_output["params"].tuning_concentration,
            experiment_output["params"].A,
            experiment_output["params"].device,
        )
    elif experiment_output["params"].input_type == "feature":
        stimuli = FeatureStimuliGenerator(
            experiment_output["params"].dims, experiment_output["params"].model, experiment_output["params"].device, experiment_output["params"].preprocess
        )
    else:
        raise ValueError(f"invalid input type {experiment_output['params'].input_type} specified")

    test_trial = Trial(experiment_output["params"].dt, experiment_output["params"].C, experiment_output["params"].eps1, test_data_gen, stimuli)
    test_trial.add_phase("prestim", 0.1, "blank", False)
    test_trial.add_phase("stim_1", 0.25, "distractor", False)
    test_trial.add_phase("stim_2", 0.25, "target", False)
    test_trial.add_phase("resp", 0.1, "blank", True)

    with torch.no_grad():
        test_batch = test_trial.run_batch(experiment_output["rnn"], test_batch_size, True)

    return test_batch

if __name__ == "__main__":
    output = run_experiment(
        input_type="feature",
        output_type="feature",
        dims=1,
        p=[2/5],
        kappas=[7.0],
        q = 0,
        tuning_concentration=3.0,
        C=0.01,
        lr=5e-4,
        interactive=True,
        batch_size=10,
        num_batches=200,
        hidden_size=200,
        test_batch_size=50
    )

    #output['test_batch'] = add_test_batch(output, 1000)
    test_output = analyze_test_batch(output)
    vizualize_test_output(test_output)

    plt.plot(output["losses"])
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()


