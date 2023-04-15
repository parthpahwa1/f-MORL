import os
import numpy as np
from util_functions import hard_update, soft_update

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

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


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DiscreteGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_preferences, action_dim, hidden_dim):
        super(DiscreteGaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_preferences, (num_inputs + num_preferences)*16)
        self.linear2a = nn.Linear((num_inputs + num_preferences)*16, (num_inputs + num_preferences)*32)
        self.linear2b = nn.Linear((num_inputs + num_preferences)*32, (num_inputs + num_preferences)*64)
        self.linear2c = nn.Linear((num_inputs + num_preferences)*64, (num_inputs + num_preferences)*32)
        self.mean_linear = nn.Linear((num_inputs + num_preferences)*32, action_dim)

        self.apply(weights_init_)

        return None

    def forward(self, state, preference):
        input = torch.cat([state, preference], 1)

        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2a(x))
        x = F.relu(self.linear2b(x))
        x = F.relu(self.linear2c(x))

        mean = self.mean_linear(x)
        return mean

    def get_probs(self, state, preference):
        mean = self.forward(state, preference)
        probs = torch.softmax(mean, dim=1)
        return probs

    def sample(self, state, preference):
        mean = self.forward(state, preference)
        probs = torch.softmax(mean, dim=1)
        m = torch.distributions.Categorical(probs)
        
        action = m.sample()
        log_prob = m.log_prob(action)

        return action.reshape(-1), log_prob, torch.argmax(probs, dim=1).reshape(-1)

    def to(self, device):
        return super(DiscreteGaussianPolicy, self).to(device)

    pass