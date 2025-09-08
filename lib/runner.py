import os
from datetime import datetime
from typing import Dict, Any

import torch
from tqdm import tqdm

from .config import ExperimentParams
from .generators import DataGenerator, SpatialStimuliGenerator, FeatureStimuliGenerator
from .rnn import RNN
from .trial import Trial
from .utils import criterion, my_loss_spatial, analyze_test_batch, visualize_test_output
from .save import save_experiment, to_cpu


def run_experiment(params: ExperimentParams) -> Dict[str, Any]:
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

    rnn_input_size = stimuli.n_inputs
    if params.output_type == "angle":
        params.output_size = 2
    elif params.output_type == "angle_color":
        params.output_size = 4
    else:
        params.output_size = rnn_input_size

    trial = Trial(params.dt, params.C, params.eps1, datagen, stimuli)
    trial.add_phase("prestim", 0.1, "blank", False)
    trial.add_phase("stim_1", 0.25, "distractor", False)
    trial.add_phase("stim_2", 0.25, "target", False)
    trial.add_phase("resp", 0.1, "blank", True)

    rnn = RNN(
        rnn_input_size,
        params.hidden_size,
        params.output_size,
        params.dt,
        params.tau,
    )
    rnn.to(params.device)

    opt = torch.optim.Adam(rnn.parameters(), params.lr)
    losses = []
    epochs = (
        tqdm(range(params.num_batches))
        if params.interactive
        else range(params.num_batches)
    )
    test_batch = None
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
                    )
                    io_vec = torch.stack(
                        [torch.cos(io_angles), torch.sin(io_angles)], dim=-1
                    ).unsqueeze(1)
                    ideal_loss = my_loss_spatial(batch["target_angles"], io_vec)
                print(f"ideal observer loss: {ideal_loss.item()}")

    if params.test_batch_size > 0:
        with torch.no_grad():
            test_batch = trial.run_batch(rnn, params.test_batch_size, True)

    return {"params": params, "rnn": rnn, "losses": losses, "test_batch": test_batch}


def run_and_save(params: ExperimentParams, out_dir: str, *, save_test_batch: bool = True, save_cnn_features: bool = False, tag: str = "") -> str:
    output = run_experiment(params)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""
    exp_dir = os.path.join(out_dir, f"exp-{ts}{suffix}")
    os.makedirs(exp_dir, exist_ok=True)

    save_experiment(output, exp_dir, save_test_batch=save_test_batch, save_cnn_features=save_cnn_features)

    if not output['test_batch'] is None:
        test_output = analyze_test_batch(output)
        torch.save(to_cpu(test_output), os.path.join(exp_dir, "test_output.pt"))
        visualize_test_output(test_output, dims=params['dims'], fname=os.path.join(exp_dir, "test_performance.png"))

    import matplotlib.pyplot as plt
    plt.close()
    plt.plot(output["losses"])
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(exp_dir, "training_loss.png"))

    return exp_dir


def add_test_batch(experiment_output: Dict[str, Any], test_batch_size: int, *, build_cnn_if_needed: bool = True) -> Dict[str, Any]:
    """Generate a test batch for an existing or loaded experiment.

    Accepts either the live training output from run_experiment or a dict
    produced by load_experiment. If the experiment used feature inputs and the
    params object lacks a CNN model (common when loading with build_cnn=False),
    this will reconstruct a full params object with the CNN from the saved config
    when build_cnn_if_needed is True.
    """
    params = experiment_output.get("params")
    cfg = experiment_output.get("config")

    # If no params or missing CNN for feature inputs, try to rebuild from config
    if (params is None or (
        getattr(params, "input_type", None) == "feature" and getattr(params, "model", None) is None
    )) and build_cnn_if_needed:
        if cfg is None:
            raise ValueError("Cannot rebuild params: missing 'config' in experiment_output")
        from .save import params_from_config  # local import to avoid cycles at module import
        params = params_from_config(cfg, build_cnn=True)

    if params is None:
        raise ValueError("experiment_output missing 'params'")

    rnn = experiment_output.get("rnn")
    if rnn is None:
        raise ValueError("experiment_output missing 'rnn'")
    # Use the RNN's existing device as the runtime device
    try:
        runtime_device = next(rnn.parameters()).device
    except StopIteration:
        runtime_device = torch.device("cpu")

    datagen = DataGenerator(
        params.dims, params.kappas, params.p, params.q, runtime_device
    )

    if params.input_type == "spatial":
        stimuli = SpatialStimuliGenerator(
            params.dims,
            params.input_size,
            params.tuning_concentration,
            params.A,
            runtime_device,
        )
    elif params.input_type == "feature":
        if getattr(params, "model", None) is None or getattr(params, "preprocess", None) is None:
            raise ValueError("Feature stimuli requested but params.model/preprocess are missing; set build_cnn_if_needed=True or load with build_cnn=True")
        # Ensure CNN is on the runtime device
        params.model.to(runtime_device)
        stimuli = FeatureStimuliGenerator(
            params.dims, params.model, runtime_device, params.preprocess
        )
    else:
        raise ValueError(f"invalid input type {params.input_type} specified")

    test_trial = Trial(params.dt, params.C, params.eps1, datagen, stimuli)
    test_trial.add_phase("prestim", 0.1, "blank", False)
    test_trial.add_phase("stim_1", 0.25, "distractor", False)
    test_trial.add_phase("stim_2", 0.25, "target", False)
    test_trial.add_phase("resp", 0.1, "blank", True)

    with torch.no_grad():
        test_batch = test_trial.run_batch(rnn, test_batch_size, True)

    return test_batch