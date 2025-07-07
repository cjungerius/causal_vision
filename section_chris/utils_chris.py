import torch
from torch import pi
from torch import Tensor as T
import torch.nn.functional as F
import numpy as np
from section_6.utils import generate_sensory_input_spatial, errors_spatial

def generate_batch_of_2d_wm_targets(n_a: int, batch_size: int):
    #return torch.randint(1, n_a+1, (batch_size,2))
    return torch.rand(batch_size,2) * n_a

def generate_batch_of_continuous_wm_targets(n_a: int, batch_size: int):
    #return torch.randint(1, n_a+1, (batch_size,2))
    return torch.rand(batch_size,) * n_a

def generate_2d_sensory_input(rs: T, fixation: bool, n_a: int):
    batch_size = rs.shape[0]
    batch = torch.zeros(batch_size, n_a, n_a)
    for i in range(batch.size(0)):
        x, y = rs[i,:]
        for j in range(batch.size(1)):
            x_dist = j-x+1
            for k in range(batch.size(2)):
                y_dist = k-y+1
                batch[i,j,k] = torch.exp((x_dist**2 + y_dist**2)/-2)
    batch = batch.view(batch_size, n_a**2)
    fixation_vector = torch.ones(batch_size) if fixation else torch.zeros(batch_size)
    return torch.concat([batch, fixation_vector.unsqueeze(1)], 1)

def generate_same_different(batch_size: int, p: float = 0.5):
    x = torch.rand((batch_size,))
    
    return torch.where(x > p, 2, 1)

def generate_correlated_binary_pairs(batch_size: int, p: float = 0.5, rho: float = 0.0):
    """
    Generate binary (x, y) pairs with:
    - x ~ Bernoulli(p)
    - y has same marginal p
    - Pearson correlation between x and y = rho

    Args:
        batch_size (int): Number of samples
        p (float): P(x=1) and P(y=1)
        rho (float): Desired Pearson correlation between x and y (-1 to 1)

    Returns:
        Tensor of shape (batch_size, 2) with values in {1, 2}
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"
    max_rho = min(p, 1-p) / (p * (1-p))
    assert -max_rho <= rho <= max_rho, f"rho must be in [-{max_rho:.2f}, {max_rho:.2f}] for p={p}"

    # Joint probabilities
    cov = rho * p * (1 - p)
    P11 = p**2 + cov
    P00 = (1 - p)**2 + cov
    P10 = p * (1 - p) - cov
    P01 = p * (1 - p) - cov

    joint_probs = torch.tensor([P00, P01, P10, P11])
    joint_probs = torch.clamp(joint_probs, 0, 1)  # Avoid numerical issues

    dist = torch.distributions.Categorical(joint_probs)
    samples = dist.sample((batch_size,))

    # Map to binary (x, y)
    pair_map = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    pairs = pair_map[samples]

    return pairs + 1  # Optional: shift {0,1} to {1,2}

def generate_same_different_2d(batch_size: int, p: float = 0.5, q: float = 0.5):
    x = torch.bernoulli(torch.full((batch_size,), 1.0-p)).int()
    y = torch.bernoulli(torch.full((batch_size,), 1.0-p)).int()
    same = torch.bernoulli(torch.full((batch_size,), q))
    y = torch.where(same==1,x,y)
    return torch.stack((x,y),1) + 1

def generate_batch_of_multivariate_inputs(batch_size: int, r: float = 0.5):
    dist = torch.distributions.MultivariateNormal(torch.zeros(2),torch.tensor([[1,r],[r,1]]))
    sample = dist.sample((batch_size,)) % (2*torch.pi)


def generate_second_sensory_input(rs: T, where_diff: T, fixation: bool, n_a: int):
    rs = rs.clone()
    batch_size = rs.shape[0]
    shift_size = torch.randint(1,n_a//2,(batch_size,))
    shift_sign = torch.randint(0,2,(batch_size,)) * 2 - 1
    shifts = shift_size * shift_sign
    rs[where_diff==2] += shifts[where_diff==2]
    rs = ((rs - 1) % n_a) + 1
    one_hots = F.one_hot(rs-1, n_a)                                                     # [batch, n_a]
    fixation_vector = torch.ones(batch_size) if fixation else torch.zeros(batch_size)   # [batch]
    return torch.concat([one_hots, fixation_vector.unsqueeze(1)], 1)   

def generate_second_sensory_input_spatial(rs: T, where_diff: T, fixation: bool, n_a: int, A: float, kappa: float):
    rs = rs.clone()
    batch_size = rs.shape[0]
    shift_size = torch.randint(1,n_a//2,(batch_size,))
    shift_sign = torch.randint(0,2,(batch_size,)) * 2 - 1
    shifts = shift_size * shift_sign
    rs[where_diff==2] += shifts[where_diff==2]
    rs = ((rs - 1) % n_a) + 1
    return generate_sensory_input_spatial(rs, fixation, n_a, A, kappa)

def loss_function_grid(rs: T, network_outputs: T, n_a: int):
    xs = (rs[:,0] / n_a) - (n_a / 2)
    ys = (rs[:,1] / n_a) - (n_a / 2)


    x_errors = (network_outputs[:,:,0].unsqueeze(-1) - xs)**2
    y_errors = (network_outputs[:,:,1].unsqueeze(-1) - ys)**2
    return (x_errors + y_errors).mean()


def errors_grid(rs: T, network_outputs: T, n_a: int):
    xs = (rs[:,0] / n_a) - (n_a / 2)
    ys = (rs[:,1] / n_a) - (n_a / 2)
    x_errors = ((network_outputs[:,:,0].unsqueeze(-1) - xs)**2).mean()
    y_errors = ((network_outputs[:,:,1].unsqueeze(-1) - ys)**2).mean()

    return x_errors, y_errors

def loss_function_grid_2(rs: T, network_outputs: T, n_a: int):

    centre = torch.median(torch.arange(1,n_a+1))

    xs = (rs[:,0] - centre)/n_a
    ys = (rs[:,1] - centre)/n_a

    target_angles = torch.arctan2(xs,ys)
    target_magnitudes = (xs**2 + ys
    **2)**0.5

    output_angles = torch.arctan2(network_outputs[:,:,0], network_outputs[:,:,1])
    output_magnitudes = (network_outputs[:,:,0]**2 + network_outputs[:,:,1]**2)**0.5

    angle_errors = target_angles - output_angles.unsqueeze(-1)

    # Small thing to normalise angle errors to [-\pi, \pi]
    k = torch.floor((pi - angle_errors)/(2 * pi))
    angle_errors = angle_errors + (k * 2 * pi)
    angle_errors = angle_errors.abs().mean()   # abs then mean to account for symmetry

    magnitude_errors = (target_magnitudes - output_magnitudes.unsqueeze(-1)).abs().mean()

    return (angle_errors + magnitude_errors).mean()


def generate_sensory_input_2d_vonmises(rs: T, fixation: bool, n_a: int, A: float, kappa: float, noise=0.0):
    """Additionally need to set the scaling and concentration parameters of the von Mises bump (assume equal for now)"""
    batch_size = rs.shape[0]

    unit_indices = torch.arange(1, n_a+1).unsqueeze(0).repeat(batch_size, 1)

    angle_diffs_x = (rs[:,0].unsqueeze(-1) - unit_indices + torch.randn(batch_size).unsqueeze(-1) * noise) * 2 * pi / n_a
    angle_diffs_y = (rs[:,1].unsqueeze(-1) - unit_indices + torch.randn(batch_size).unsqueeze(-1) * noise) * 2 * pi / n_a

    bumps_x = A * (kappa * angle_diffs_x.cos()).exp().unsqueeze(2)                                    # [batch, n_a]
    bumps_y = A * (kappa * angle_diffs_y.cos()).exp().unsqueeze(1)

    bumps = bumps_x * bumps_y / torch.exp(torch.tensor(kappa))

    bumps = bumps.view(batch_size,-1)

    fixation_vector = torch.ones(batch_size) if fixation else torch.zeros(batch_size)   # [batch]
    return torch.cat([bumps, fixation_vector.unsqueeze(1)], 1)                       # [batch, n_a + 1]

def loss_function_2d_vonmises(rs: T, network_outputs: T, n_a: int):
    angles = rs * 2 * pi / n_a  # shape: (batch, 2)
    true_vectors = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (batch, 2, 2)

    # network_outputs: (batch, N, 4)
    # split into two (cos, sin) pairs
    pred_1 = network_outputs[:, :, 0:2]  # (batch, N, 2)
    pred_2 = network_outputs[:, :, 2:4]

    target_1 = true_vectors[:, 0, :].unsqueeze(1)  # (batch, 1, 2)
    target_2 = true_vectors[:, 1, :].unsqueeze(1)

    loss_1 = ((pred_1 - target_1) ** 2).sum(dim=2)  # sum over x/y
    loss_2 = ((pred_2 - target_2) ** 2).sum(dim=2)

    return (loss_1 + loss_2).mean()

def errors_spatial_2d(rs: T, network_outputs: T, n_a: int):
    angle_errors_1, magnitudes_errors_1 = errors_spatial(rs[:,0], network_outputs[:,0:2], n_a)
    angle_errors_2, magnitudes_errors_2 = errors_spatial(rs[:,1], network_outputs[:,2:4], n_a)
    return (angle_errors_1 + angle_errors_2).mean(), (magnitudes_errors_1 + magnitudes_errors_2).mean()


def loss_function_cosine_similarity(rs: T, network_outputs: T, n_a: int):
    """
    Computes the cosine similarity loss between the true and predicted vectors.
    rs: (batch, N) - true vectors
    network_outputs: (batch, N, 2) - predicted vectors
    """
    angles = rs * 2 * pi / n_a  # shape: (batch, N)
    true_vectors = torch.stack([angles.cos(), angles.sin()], dim=-1)  # (batch, N, 2)

    # network_outputs: (batch, N, 2)
    cos_sim = F.cosine_similarity(network_outputs, true_vectors, dim=-1)  # (batch, N)

    # We want to maximize cosine similarity, so we minimize the negative
    return -cos_sim.mean()  # Mean over batch and N
# %%
def loss_function_spatial_non_recurrent(rs: T, network_outputs: T, n_a: int):
    angles = rs * 2 * pi / n_a
    xs = angles.cos()
    ys = angles.sin()
    x_errors = (network_outputs[:,0] - xs)**2
    y_errors = (network_outputs[:,1] - ys)**2
    return (x_errors + y_errors).mean()