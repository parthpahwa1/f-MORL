import math
import torch
import itertools
import numpy as np
import pandas as pd
import time 
from tqdm import tqdm
from pymoo.indicators.hv import HV

from typing import List, Optional, Tuple, Union

if  torch.cuda.is_available():
    device = torch.device("cuda")
    FloatTensor = torch.cuda.FloatTensor 
    LongTensor = torch.cuda.LongTensor 
    ByteTensor = torch.cuda.ByteTensor 
    Tensor = FloatTensor
else:
    device = torch.device("cpu")
    FloatTensor = torch.FloatTensor 
    LongTensor = torch.LongTensor 
    ByteTensor = torch.ByteTensor 
    Tensor = FloatTensor

def logistic(x, scaling):
    return 1 / (1+np.exp(-x/scaling))

def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point.
    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors
    Returns:
        float: Hypervolume metric
    """
    ind = HV(ref_point=ref_point*-1)
    return ind(np.array(points)*-1)


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def generate_next_preference(preference, alpha=0.2):
    preference = np.array(preference)
    preference += 1e-6
    
    return FloatTensor(np.random.dirichlet(alpha*preference))

def generate_next_preference_gaussian(preference, alpha=0.2):
    
    cov = np.identity(preference.shape[0])*0.000001*alpha
    new_next_preference = np.random.multivariate_normal(preference, cov, 1)[0]
    new_next_preference[new_next_preference < 0] = 0
    new_next_preference += 1e-9
    new_next_preference = new_next_preference/np.sum(new_next_preference)
    
    return FloatTensor(new_next_preference)


def find_in(A: np.ndarray, B: np.ndarray, eps: float = 0.2) -> Tuple[float, float]:
    """
    Find elements of A in B with a tolerance of relative error of eps.

    Parameters:
    A (numpy.ndarray): First array of elements to be found
    B (numpy.ndarray): Second array of elements to be searched in
    eps (float, optional): Tolerance of relative error (default is 0.2)

    Returns:
    Tuple (float, float): Count of elements in A found in B and count of elements in B found in A
    """
    cnt1, cnt2 = 0.0, 0.0

    for a in A:
        for b in B:
            if eps > 0.001:
                if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                    cnt1 += 1.0
                    break
            else:
                if np.linalg.norm(a - b, ord=1) < 0.5:
                    cnt1 += 1.0
                    break

    for b in B:
        for a in A:
            if eps > 0.001:
                if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                    cnt2 += 1.0
                    break
            else:
                if np.linalg.norm(a - b, ord=1) < 0.5:
                    cnt2 += 1.0
                    break

    return cnt1, cnt2
