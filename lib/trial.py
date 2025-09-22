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

    def run_batch(
        self,
        model: RNN,
        batch_size: int,
        test: bool = False,
        max_chunk_size: int | None = 2000,
        **datagen_kwargs,
    ):
        """
        Run a batch (optionally in memory-saving chunks) through all phases.

        Args:
            model: RNN model with step() and init_hidden().
            batch_size: Requested batch size from datagen.
            test: Passed through to datagen / stimuli.
            max_chunk_size: If set and batch exceeds this, process sequential chunks
                            to lower peak memory.
            **datagen_kwargs: Additional arguments to data generator.

        Returns:
            batch dict with 'outputs' key: (B, T_out, output_dim) or None.
        """
        # Derive sizes
        N = getattr(model, "hidden_size", None) or model.init_hidden(1).shape[1]
        n_in = getattr(model, "input_size", None) or self.stimuli.n_inputs
        device = next(model.parameters()).device

        batch = self.datagen(batch_size, test, **datagen_kwargs)

        # Determine true batch size
        if "target_angles" in batch:
            true_batch_size = batch["target_angles"].shape[0]
        elif "target_colors" in batch:
            true_batch_size = batch["target_colors"].shape[0]
        else:
            raise ValueError("Cannot determine batch size from batch dict.")

        if (max_chunk_size is None) or (true_batch_size <= max_chunk_size):
            target_inputs, distractor_inputs = self.stimuli(batch, test)
            eta_tilde = torch.randn(true_batch_size, N, device=device)
            hidden = model.init_hidden(true_batch_size)
            empty_inputs = torch.zeros(true_batch_size, n_in, device=device)
            outputs = []
            for phase in self.phases:
                if phase.type == "target":
                    inputs = target_inputs
                elif phase.type == "distractor":
                    inputs = distractor_inputs
                else:
                    inputs = empty_inputs
                for _ in range(phase.steps):
                    eta_tilde = self.eps1 * eta_tilde + self.eps2 * torch.randn(
                        true_batch_size, N, device=device
                    )
                    eta = eta_tilde * self.C
                    if not phase.output:
                        hidden = model.step(hidden, inputs, eta, False)
                    else:
                        hidden, output = model.step(hidden, inputs, eta, True)
                        outputs.append(output)
            batch["outputs"] = torch.stack(outputs, dim=1) if outputs else None
            return batch

        # Helper to slice batch dict
        def _slice_batch(b, sl, B):
            return {
                k: (v[sl] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == B else v)
                for k, v in b.items()
            }

        chunk_outputs = []
        for start in range(0, true_batch_size, max_chunk_size):
            end = min(start + max_chunk_size, true_batch_size)
            sl = slice(start, end)
            chunk_batch_size = end - start

            chunk_batch = _slice_batch(batch, sl, true_batch_size)
            t_inputs_chunk, d_inputs_chunk = self.stimuli(chunk_batch, test)
            empty_inputs = torch.zeros(chunk_batch_size, n_in, device=device)

            eta_tilde = torch.randn(chunk_batch_size, N, device=device)
            hidden = model.init_hidden(chunk_batch_size)

            this_chunk_outputs = []
            for phase in self.phases:
                if phase.type == "target":
                    inputs = t_inputs_chunk
                elif phase.type == "distractor":
                    inputs = d_inputs_chunk
                else:
                    inputs = empty_inputs
                for _ in range(phase.steps):
                    eta_tilde = self.eps1 * eta_tilde + self.eps2 * torch.randn(
                        chunk_batch_size, N, device=device
                    )
                    eta = eta_tilde * self.C
                    if not phase.output:
                        hidden = model.step(hidden, inputs, eta, False)
                    else:
                        hidden, output = model.step(hidden, inputs, eta, True)
                        this_chunk_outputs.append(output)

            if this_chunk_outputs:
                chunk_outputs.append(torch.stack(this_chunk_outputs, dim=1))

        batch["outputs"] = torch.cat(chunk_outputs, dim=0) if chunk_outputs else None
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
