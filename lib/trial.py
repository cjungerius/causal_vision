# %%
import torch
from .rnn import RNN
from .generators import (
    DataGenerator,
    SpatialStimuliGenerator,
    FeatureStimuliGenerator,
)


class Trial:
    def __init__(
        self,
        dt: float,
        C: float,
        eps1: float,
        datagen: DataGenerator,
        stimuli: SpatialStimuliGenerator | FeatureStimuliGenerator,
    ):
        self.dt = dt
        self.eps1 = eps1
        self.eps2 = (1 - self.eps1**2) ** 0.5
        self.C = C
        self.phases = []
        self.datagen = datagen
        self.stimuli = stimuli

    def add_phase(self, name, T, type, output):
        self.phases.append(Phase(name, T, self.dt, type, output))

    def remove_phase(self, i=-1):
        self.phases.pop(i)

    def clear_phases(self):
        self.phases = []

    def run_batch(self, model: RNN, batch_size, test=False):
        # derive sizes
        N = getattr(model, "hidden_size", None) or model.init_hidden(1).shape[1]
        n_in = getattr(model, "input_size", None) or self.stimuli.n_inputs

        # get model device to keep the 0 input on the same device
        device = next(model.parameters()).device
        # initialize
        batch = self.datagen(batch_size, test)
        target_inputs, distractor_inputs = self.stimuli(batch, test)
        eta_tilde = torch.randn(batch_size, N).to(device)
        hidden = model.init_hidden(batch_size)
        empty_inputs = torch.zeros(batch_size, n_in, device=device).float()
        outputs = []

        # go through phases
        for phase in self.phases:
            if phase.type == "target":
                inputs = target_inputs
            elif phase.type == "distractor":
                inputs = distractor_inputs
            else:
                inputs = empty_inputs

            for _ in range(phase.steps):
                eta_tilde = self.eps1 * eta_tilde + self.eps2 * torch.randn(
                    batch_size, N, device=device
                )
                eta = eta_tilde * self.C
                if not phase.output:
                    hidden = model.step(hidden, inputs, eta, False)
                else:
                    hidden, output = model.step(hidden, inputs, eta, True)
                    outputs.append(output)

        batch["outputs"] = torch.stack(outputs, dim=1) if len(outputs) > 0 else None
        return batch


class Phase:
    def __init__(self, name: str, T: float, dt: float, type=str, output=False):
        self.name = name
        self.output = output
        self.steps = int(T / dt)
        self.type = type


# T_prestim = 0.1     # in seconds
# T_stim_1 = 0.25
# T_stim_2 = 0.25
# T_delay = 0.0
# T_resp = 0.1

# dt = 0.1

# prestim_timesteps = int(T_prestim / dt)
# stim_timesteps_1 = int(T_stim_1 / dt)
# stim_timesteps_2 = int(T_stim_2 / dt)
# delay_timesteps = int(T_delay / dt)
# resp_timesteps = int(T_resp / dt)

# self.trial_params = {
#     "prestim_timesteps": prestim_timesteps,
#     "stim_timesteps_1": stim_timesteps_1,
#     "stim_timesteps_2": stim_timesteps_2,
#     "delay_timesteps": delay_timesteps,
#     "resp_timesteps": resp_timesteps
# }
# %%
