from torch import nn, randn
from torch import Tensor as T
import torch.nn.functional as F
from typing import overload, Literal, Tuple

class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dt: float, tau: float):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)
        self.tau = tau
        self.dt = dt
        self.alpha = self.dt / self.tau
        # No internal hidden state

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return randn(batch_size, self.hidden_size, device=device)

    @overload
    def step(self, hidden: T, input: T, noise: T, return_output: Literal[False] = ...) -> T: ...
    @overload
    def step(self, hidden: T, input: T, noise: T, return_output: Literal[True]) -> Tuple[T, T]: ...

    def step(self, hidden: T, input: T, noise: T, return_output: bool=False):
        leaky_term = (1 - self.alpha) * hidden
        weighted_rates = self.W(F.relu(hidden))
        input_projection = self.W_in(input)
        integration_term = self.alpha * (weighted_rates + noise + input_projection)
        hidden = leaky_term + integration_term
        if return_output:
            output = self.W_out(F.relu(hidden))
            return hidden, output
        else:
            return hidden