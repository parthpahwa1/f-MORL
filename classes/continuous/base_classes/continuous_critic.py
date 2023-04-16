import os
import numpy as np
from util_functions import hard_update, soft_update

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from .continuous_params import ContinuousParameters
LOG_SIG_MAX = ContinuousParameters.get_log_sig_max()
LOG_SIG_MIN = ContinuousParameters.get_log_sig_min()
VALUE_SCALING = ContinuousParameters.get_value_scaling()
epsilon = ContinuousParameters.get_epsilon()

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

class Continuous_G_Network(nn.Module):
    def __init__(self, num_inputs, num_preferences, action_dim, hidden_dim):
        super(Continuous_G_Network, self).__init__()
        # self.batch_norm = nn.BatchNorm1d(num_inputs + num_preferences + action_dim)
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_preferences + action_dim, hidden_dim)
        self.linear2a = nn.Linear(hidden_dim, hidden_dim)
        self.linear2b = nn.Linear(hidden_dim, hidden_dim)
        # self.linear2c = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear3= nn.Linear(num_inputs + num_preferences + action_dim, hidden_dim)
        self.linear4a = nn.Linear(hidden_dim, hidden_dim)
        self.linear4b = nn.Linear(hidden_dim, hidden_dim)
        # self.linear4c = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
        return None

    def forward(self, state, preference, action):
        xu = torch.cat([state, preference, action], 1)
        # xu = self.batch_norm(xu)
        
        x1 = F.relu(self.linear1(xu)) 
        x1 = F.relu(self.linear2a(x1)) 
        x1 = F.relu(self.linear2b(x1)) 
        # x1 = F.relu(self.linear2c(x1)) 
        x1 = F.sigmoid(self.mean_linear1(x1)/VALUE_SCALING)
        
        x2 = F.relu(self.linear3(xu))
        x2 = F.relu(self.linear4a(x2)) 
        x2 = F.relu(self.linear4b(x2)) 
        # x2 = F.relu(self.linear4c(x2)) 
        x2 = F.sigmoid(self.mean_linear2(x2)/VALUE_SCALING)

        return x1, x2

