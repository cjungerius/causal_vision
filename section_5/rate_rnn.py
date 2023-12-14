from torch.nn import Module, Parameter
from torch import Tensor as T
from torch import randn
from torch.nn.functional import relu

class ReluRateRNN(Module):
    "No input term yet, and noise provided externally"

    def __init__(self, dt: float, num_neurons: int, tau_m: float):
        super(ReluRateRNN, self).__init__()
        
        self.dt = dt
        self.tau_m = tau_m
        self.alpha = self.dt / self.tau_m

        # Recurrent synaptic weights
        W = randn(num_neurons, num_neurons)
        W = Parameter(W)
        self.register_parameter(name = 'W', param = W)

        # Keep track of membrane voltage
        self.u = randn(num_neurons)

    def non_linearity(self, voltages: T) -> T:
        return relu(voltages)

    def get_recurrence_term(self, voltages: T) -> T:
        return self.W @ self.non_linearity(voltages)
    
    def step_dynamics(self, noise_term: T) -> T:
        "Updates the self.u property. Also returns it for good measure"
        leaky_term = (1 - self.alpha) * self.u

        weighted_rates = self.get_recurrence_term(self.u)
        integration_term = self.alpha * (weighted_rates + noise_term)
        self.u = leaky_term + integration_term

        return self.u
