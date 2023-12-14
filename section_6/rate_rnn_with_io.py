from torch.nn import Module, Parameter
from torch import Tensor as T
from torch import randn
from torch.nn.functional import relu

class ReluRateRNNWithIO(Module):
    "Now with input/output projections. Batch size has to be defined unfortunately!"

    def __init__(self, dt: float, num_neurons: int, input_size: int, output_size: int, tau_m: float):
        super(ReluRateRNNWithIO, self).__init__()
        
        self.dt = dt
        self.tau_m = tau_m
        self.alpha = self.dt / self.tau_m

        # Recurrent synaptic weights
        W = randn(num_neurons, num_neurons) / num_neurons
        W = Parameter(W)
        self.register_parameter(name = 'W', param = W)

        # Input weights
        W_in = randn(input_size, num_neurons) / num_neurons**0.5
        W_in = Parameter(W_in)
        self.register_parameter(name = 'W_in', param = W_in)
    
        # Output weights
        W_out = randn(num_neurons, output_size) / num_neurons**0.5
        W_out = Parameter(W_out)
        self.register_parameter(name = 'W_out', param = W_out)

        self.num_neurons = num_neurons

    def initialise_u(self, batch_size):
        self.u = randn(batch_size, self.num_neurons)

    def non_linearity(self, voltages: T) -> T:
        return relu(voltages)

    def get_recurrence_term(self, voltages: T) -> T:
        return (self.non_linearity(voltages) @ self.W)  # TODO: double check whether this needs to be transposed!
    
    def step_dynamics(self, input_vector: T, noise_term: T, return_output: bool) -> T:
        """Updates the self.u property and return it. Only if we want it, also projects to output space via nonlinearity"""
        leaky_term = (1 - self.alpha) * self.u

        weighted_rates = self.get_recurrence_term(self.u)
        input_projection = (input_vector @ self.W_in)
        integration_term = self.alpha * (weighted_rates + noise_term + input_projection)

        self.u = leaky_term + integration_term

        if return_output:
            r = self.non_linearity(self.u)
            output = (r @ self.W_out)
            return self.u, output
        else:
            return self.u
