import torch
from torch import pi
from torch import Tensor as T
import torch.nn.functional as F
import numpy as np
from section_6.utils import generate_sensory_input_spatial

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

def generate_grid_sensory_input(rs: T, fixation: bool, n_a: int, sd: float):
    side = (n_a**.5)
    if not side.is_integer():
        raise Exception("chosen value for n_a doesn't allow for a square view!")
    
    inputs = torch.ones((rs.size(0), n_a))

    