import torch
from torch import nn 
import torch.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class FNetwork(nn.Module):
    def __init__(self, num_inputs, num_preferences, hidden_dim):
        super(FNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs+num_preferences, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2a = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        return None

    def forward(self, state, preference):
        input = torch.cat([state, preference], 1)
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2a(x))
        x = self.linear3(x)
        return x


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, num_preferences, hidden_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_preferences + 1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_a = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_preferences + 1, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_a = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        return None

    def forward(self, state, preference, action):
        xu = torch.cat([state, preference, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear2_a(x1))
        x1 = self.linear3(x1)
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = F.relu(self.linear5_a(x2))
        x2 = self.linear6(x2)

        return x1, x2


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, num_preferences, hidden_dim, action_space=None):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_preferences, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        # if action_space is None:
        #     self.action_scale = torch.tensor(1.)
        #     self.action_bias = torch.tensor(0.)
        # else:
        #     self.action_scale = torch.FloatTensor(
        #         0)
        #     self.action_bias = torch.FloatTensor(
        #         0)

        return None

    def forward(self, state, preference):
        input = torch.cat([state, preference], 1)
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
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
        # self.action_scale = self.action_scale.to(device)
        # self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)
