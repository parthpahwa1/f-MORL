import os
import numpy as np
from util_functions import hard_update, soft_update

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

torch.autograd.set_detect_anomaly(True)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
VALUE_SCALING = 1000
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

class Continuous_F_Network(nn.Module):
    def __init__(self, num_inputs, num_preferences, hidden_dim):
        super(Continuous_F_Network, self).__init__()

        # self.batch_norm = nn.BatchNorm1d(num_inputs + num_preferences)
        self.linear1 = nn.Linear(num_inputs + num_preferences, hidden_dim)
        self.linear2a = nn.Linear(hidden_dim, hidden_dim)
        self.linear2b = nn.Linear(hidden_dim, hidden_dim)
        # self.linear2c = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear1 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, preference):
        input = torch.cat([state, preference], 1)
        # input = self.batch_norm(input)
        
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2a(x)) 
        x = F.relu(self.linear2b(x)) 
        # x = F.relu(self.linear2c(x)) + x
        x = F.sigmoid(self.mean_linear1(x)/VALUE_SCALING)

        return x
