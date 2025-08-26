import torch
from rnn import RNN

class Trial():
    def __init__(self, phases: list, Ts: list, dt: float, C: float, eps1: float):
        
        self.dt = dt
        self.eps1 = eps1
        self.eps2 = (1 - self.eps1**2) ** 0.5
        self.C = C
    
        
        self.phases = []        ## Define simulation parameters

        T_prestim = 0.1     # in seconds
        T_stim_1 = 0.25
        T_stim_2 = 0.25
        T_delay = 0.0
        T_resp = 0.1
        
        dt = 0.1

        prestim_timesteps = int(T_prestim / dt)
        stim_timesteps_1 = int(T_stim_1 / dt)
        stim_timesteps_2 = int(T_stim_2 / dt)
        delay_timesteps = int(T_delay / dt)
        resp_timesteps = int(T_resp / dt)
    
        self.trial_params = {
            "prestim_timesteps": prestim_timesteps,
            "stim_timesteps_1": stim_timesteps_1,
            "stim_timesteps_2": stim_timesteps_2,
            "delay_timesteps": delay_timesteps,
            "resp_timesteps": resp_timesteps
        }
        
    def add_phase(self, T, input, output):
        self.phases.append(Phase(T, self.dt, input, output))
        
    def remove_phase(self, i=-1):
        self.phases.pop(i)
        
    def run_batch(self, model: RNN, batch_size, N, n_a):
        
        # initialize
        batch = t.generate_batch(batch_size)
        target_inputs, distractor_inputs = t.generate_inputs(batch)
        eta_tilde = torch.randn(batch_size, N)
        hidden = model.init_hidden(batch_size)
        inputs = torch.zeros(batch_size, n_a).float()
        outputs = []
        
        #go through phases
        for phase in self.phases:
            if phase.type == "target":
                inputs = target_inputs
            elif phase.type == "distractor":
                inputs = distractor_inputs
            else:
                inputs = torch.zeros(batch_size, n_a).float()
            
            for _ in range(phase.steps):
                eta_tilde = (self.eps1  *eta_tilde + self.eps2 * torch.randn(batch_size, N))
                eta = eta_tilde * self.C

                if not phase.output:
                    hidden = model.step(hidden, inputs, eta, False)
                else:
                    hidden, output = model.step(hidden, inputs, eta, True)
                    outputs.append(output)
        
        batch["outputs"] = outputs
        
        return batch

class Phase():
    def __init__(self, T: float, dt: float, type = str, output = False):

        self.output = output
        self.steps = int(T/dt)
        self.type = type
        