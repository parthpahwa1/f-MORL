import os
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from ...parameters import ContinuousParameters

LOG_SIG_MAX = ContinuousParameters.get_log_sig_max()
LOG_SIG_MIN = ContinuousParameters.get_log_sig_min()
VALUE_SCALING = ContinuousParameters.get_value_scaling()
epsilon = ContinuousParameters.get_epsilon()

torch.autograd.set_detect_anomaly(True)

# Determine the device to use
if torch.cuda.is_available():
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


def weights_init_(m: nn.Module) -> None:
    """Initialize the weights of a model.

    Args:
        m: A model whose weights to initialize.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ContinuousGaussianPolicy(nn.Module):
    def __init__(self, num_inputs: int, num_preferences: int, action_dim: int, hidden_dim: int) -> None:
        super(ContinuousGaussianPolicy, self).__init__()
        
        # self.batch_norm = nn.BatchNorm1d(num_inputs + num_preferences)
        self.linear1 = nn.Linear(num_inputs + num_preferences, hidden_dim)
        self.linear2a = nn.Linear(hidden_dim, hidden_dim)
        self.linear2b = nn.Linear(hidden_dim, hidden_dim)
        # self.linear2c = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)

        self.linear3 = nn.Linear(num_inputs + num_preferences, hidden_dim)
        self.linear4a = nn.Linear(hidden_dim, hidden_dim)
        self.linear4b = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4c = nn.Linear(hidden_dim, hidden_dim)
        self.std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state: Tensor, preference: Tensor) -> Tuple[Tensor, Tensor]:
        input = torch.cat([state, preference], 1)

        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2a(x)) 
        x = F.relu(self.linear2b(x))
        mean = F.hardtanh(self.mean_linear(x), -1, 1)

        x = F.relu(self.linear3(input))
        x = F.relu(self.linear4a(x)) 
        x = F.relu(self.linear4b(x)) 
        log_std = F.hardtanh(self.std_linear(x), LOG_SIG_MIN, LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state: Tensor, preference: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        mean, log_std = self.forward(state, preference)

        std = log_std.exp()

        m = torch.distributions.Normal(loc=mean, scale=std)
        
        x_t = m.rsample()
        action = torch.tanh(x_t)

        log_prob = m.log_prob(x_t)        
        
        return action, log_prob

    def to(self, device: str) -> 'ContinuousGaussianPolicy':
        return super(ContinuousGaussianPolicy, self).to(device)

    pass