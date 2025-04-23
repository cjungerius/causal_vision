import torch
from torch import pi
from torch import Tensor as T
import torch.nn.functional as F
import numpy as np
from section_6.utils import generate_sensory_input_spatial, errors_spatial

def generate_batch_of_2d_wm_targets(n_a: int, batch_size: int):
    return torch.randint(1, n_a+1, (batch_size,2))

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

def generate_same_different(batch_size: int):
    return torch.randint(1,3,(batch_size,))

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


def generate_sensory_input_2d_vonmises(rs: T, fixation: bool, n_a: int, A: float, kappa: float):
    """Additionally need to set the scaling and concentration parameters of the von Mises bump (assume equal for now)"""
    batch_size = rs.shape[0]

    unit_indices = torch.arange(1, n_a+1).unsqueeze(0).repeat(batch_size, 1)

    angle_diffs_x = (rs[:,0].unsqueeze(-1) - unit_indices) * 2 * pi / n_a
    angle_diffs_y = (rs[:,1].unsqueeze(-1) - unit_indices) * 2 * pi / n_a

    bumps_x = A * (kappa * angle_diffs_x.cos()).exp().unsqueeze(2)                                    # [batch, n_a]
    bumps_y = A * (kappa * angle_diffs_y.cos()).exp().unsqueeze(1)

    bumps = bumps_x * bumps_y

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

