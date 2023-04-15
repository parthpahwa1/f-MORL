import os
import numpy as np
from utils import hard_update, soft_update

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


class Discrete_G_Network(nn.Module):
    def __init__(self, num_inputs, num_preferences, action_dim, hidden_dim):
        super(Discrete_G_Network, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_preferences, (num_inputs + num_preferences)*16)
        self.linear2a = nn.Linear((num_inputs + num_preferences)*16, (num_inputs + num_preferences)*32)
        self.linear2b = nn.Linear((num_inputs + num_preferences)*32, (num_inputs + num_preferences)*64)
        self.linear2c = nn.Linear((num_inputs + num_preferences)*64, (num_inputs + num_preferences)*32)
        self.mean_linear1 = nn.Linear((num_inputs + num_preferences)*32, action_dim)

        # Q2 architecture
        self.linear3= nn.Linear(num_inputs + num_preferences, (num_inputs + num_preferences)*16)
        self.linear4a = nn.Linear((num_inputs + num_preferences)*16, (num_inputs + num_preferences)*32)
        self.linear4b = nn.Linear((num_inputs + num_preferences)*32, (num_inputs + num_preferences)*64)
        self.linear4c = nn.Linear((num_inputs + num_preferences)*64, (num_inputs + num_preferences)*32)
        self.mean_linear2 = nn.Linear((num_inputs + num_preferences)*32, action_dim)

        self.apply(weights_init_)
        return None

    def forward(self, state, preference):
        xu = torch.cat([state, preference], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2a(x1))
        x1 = F.relu(self.linear2b(x1))
        x1 = F.relu(self.linear2c(x1))
        x1 = self.mean_linear1(x1)
        
        x2 = F.relu(self.linear3(xu))
        x2 = F.relu(self.linear4a(x2))
        x2 = F.relu(self.linear4b(x2))
        x2 = F.relu(self.linear4c(x2))
        x2 = self.mean_linear2(x2)

        return x1, x2

