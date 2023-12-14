import torch
from torch import pi
from torch import Tensor as T
import torch.nn.functional as F


def generate_batch_of_wm_targets(n_a: int, batch_size: int):
    """
    Returns of batch of 'r', i.e. an index for the units around the ring
    Use 1-indexing throughout, so that the notation matches up with the handout
    """
    return torch.randint(1, n_a, (batch_size,))  # [batch]


def generate_sensory_input_non_spatial(rs: T, fixation: bool, n_a: int):
    """Generates x vector, with batch size implicit in shape of rs (batch_size)"""
    one_hots = F.one_hot(rs, n_a)                                                       # [batch, n_a]
    batch_size = rs.shape[0]
    fixation_vector = torch.ones(batch_size) if fixation else torch.zeros(batch_size)   # [batch]
    return torch.concat([one_hots, fixation_vector.unsqueeze(1)], 1)                    # [batch, n_a + 1]


def generate_sensory_input_spatial(rs: T, fixation: bool, n_a: int, A: float, kappa: float):
    """Additionally need to set the scaling and concentration parameters of the von Mises bump"""
    batch_size = rs.shape[0]
    unit_indices = torch.arange(1, n_a + 1).unsqueeze(0).repeat(batch_size, 1)              # [batch, n_a]
    angle_diffs = (rs.unsqueeze(1) - unit_indices) * 2 * pi / n_a                       # [batch, n_a]
    bumps = A * (kappa * angle_diffs.cos()).exp()                                       # [batch, n_a]
    fixation_vector = torch.ones(batch_size) if fixation else torch.zeros(batch_size)   # [batch]
    return torch.concat([bumps, fixation_vector.unsqueeze(1)], 1)                       # [batch, n_a + 1]


def generate_blank_sensory_input(n_a: int, fixation: bool, batch_size: int):
    sensory_input = torch.zeros(batch_size, n_a + 1).float()
    if fixation: 
        sensory_input[:,-1] = 1.0
    return sensory_input


def loss_function_non_spatial(rs: T, network_outputs: T):
    """
    A smarter, more compute efficient version of cross entropy loss
    network_outputs of shape [batch, resp_timesteps, n_a]
    """
    indices = rs - 1    # [batch]   convert to 0-indexing
    exp_grid = network_outputs.exp()
    softmax_denom = exp_grid.sum(-1)                            # [batch, resp_timesteps]
    softmax_num = torch.stack([exp_grid_bi[:,bi] for bi, exp_grid_bi in zip(indices, exp_grid)], 0) # [batch, resp_timesteps]
    cross_entropy_grid = (softmax_denom / softmax_num).log()  # [batch, resp_timesteps]
    return cross_entropy_grid.mean()                            # mean over both time and batch!

    
def accuracy_non_spatial(rs: T, network_outputs: T):
    """
    Categorical accuracy, averaged over batch items and time
    """
    indices = rs - 1    # [batch]   convert to 0-indexing
    estimates = network_outputs.argmax(-1)  # [batch, time]
    return (indices.unsqueeze(-1) == estimates).float().mean()
    

def loss_function_spatial(rs: T, network_outputs: T, n_a: int):
    angles = rs * 2 * pi / n_a
    xs = angles.cos().unsqueeze(-1)
    ys = angles.sin().unsqueeze(-1)
    x_errors = (network_outputs[:,:,0] - xs)**2
    y_errors = (network_outputs[:,:,1] - ys)**2
    return (x_errors + y_errors).mean()


def errors_spatial(rs: T, network_outputs: T, n_a: int):
    angles = (rs * 2 * pi / n_a).unsqueeze(-1)
    output_angles = torch.arctan2(network_outputs[:,:,0], network_outputs[:,:,1])
    angle_errors = (angles - output_angles)

    # Small thing to normalise angle errors to [-\pi, \pi]
    k = torch.floor((pi - angle_errors)/(2 * pi))
    angle_errors = angle_errors + (k * 2 * pi)
    angle_errors = angle_errors.abs().mean()   # abs then mean to account for symmetry

    output_magnitudes = (network_outputs[:,:,0]**2 + network_outputs[:,:,1]**2)**0.5
    magnitudes_errors = (output_magnitudes - 1).abs().mean()
    return angle_errors, magnitudes_errors
