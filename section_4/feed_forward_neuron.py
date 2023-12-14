from torch.nn import Module, Parameter
from torch import Tensor as T
from torch import randn

class PostSynapticNeuron(Module):
    """
    This is the single postsynaptic neuron of our feedforward network.
    At each timestep, it takes in a set of input rates from the N
        presynaptic neurons, and runs the leaky integration dynamics
    """

    def __init__(self, dt: float, num_inputs: int, tau_m: float):
        super(PostSynapticNeuron, self).__init__()
        
        self.dt = dt
        self.tau_m = tau_m
        self.alpha = self.dt / self.tau_m

        # Input synaptic weights
        w = randn(num_inputs)
        w = Parameter(w)
        self.register_parameter(name = 'w', param = w)

        # Keep track of membrane voltage
        self.v_m = randn(1)

    def get_input_voltage(self, input_rates: T) -> T:
        return self.w @ input_rates
    
    def step_dynamics(self, input_rates: T) -> T:
        "Updates the self.v_m property. Also returns it for good measure"
        weighted_rates = self.get_input_voltage(input_rates)
        leaky_term = (1 - self.alpha) * self.v_m
        integration_term = self.alpha * weighted_rates
        self.v_m = leaky_term + integration_term
        return self.v_m, weighted_rates
