import os
import numpy as np
from utils import hard_update, soft_update

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

torch.autograd.set_detect_anomaly(True)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
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
        x = F.sigmoid(self.mean_linear1(x))

        return x


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
        x1 = F.sigmoid(self.mean_linear1(x1))
        
        x2 = F.relu(self.linear3(xu))
        x2 = F.relu(self.linear4a(x2)) 
        x2 = F.relu(self.linear4b(x2)) 
        # x2 = F.relu(self.linear4c(x2)) 
        x2 = F.sigmoid(self.mean_linear2(x2))

        return x1, x2


class ContinuousGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_preferences, action_dim, hidden_dim):
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

        return None

    def forward(self, state, preference):
        input = torch.cat([state, preference], 1)
        # input = self.batch_norm(input)
        # Mean network forward
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2a(x)) 
        x = F.relu(self.linear2b(x))
        # x = F.relu(self.linear2c(x)) +x
        mean = F.hardtanh(self.mean_linear(x), -1, 1)

        # Standard deviation forward
        x = F.relu(self.linear3(input))
        x = F.relu(self.linear4a(x)) 
        x = F.relu(self.linear4b(x)) 
        # x = F.relu(self.linear4c(x)) +x
        log_std = F.hardtanh(self.std_linear(x), LOG_SIG_MIN, LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state, preference):
        self.eval()
        mean, log_std = self.forward(state, preference)

        std = log_std.exp()

        # Add std for distribution init
        m = torch.distributions.Normal(loc=mean, scale=std)
        
        x_t = m.rsample()
        action = torch.tanh(x_t)

        log_prob = m.log_prob(x_t)        
        
        return action, log_prob

    def to(self, device):
        return super(ContinuousGaussianPolicy, self).to(device)


class ContinuousSAC(object):

    def __init__(self, num_inputs, args):
        super().__init__()

        self.num_inputs = num_inputs

        self.args = args

        self.rng = np.random.RandomState(args.seed)

        self.action_space = args.action_space
        self.num_actions = args.action_dim
        self.action_dim = args.action_dim
        self.n_preferences = args.num_preferences
        self.n_weights = args.num_weights

        self.target_update_interval = args.target_update_interval

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.i_episode = 0
        
        # device = ""
        # if args.cuda:
        #     device = "cuda"
        # elif args.mps:
        #     device = "mps"
        # else:
        #     device = "cpu"

        self.device = torch.device(device)

        self.critic = Continuous_G_Network(self.num_inputs, self.n_preferences, self.action_dim, args.hidden_size).to(device)
        self.critic_optim = Adam(self.critic.parameters())

        self.critic_target = Continuous_G_Network(self.num_inputs, self.n_preferences, self.action_dim, args.hidden_size).to(device)

        hard_update(self.critic_target, self.critic)

        self.f_critic = Continuous_F_Network(self.num_inputs, self.n_preferences, args.hidden_size).to(device)
        self.f_optim = Adam(self.f_critic.parameters())

        self.f_target = Continuous_F_Network(self.num_inputs, self.n_preferences, args.hidden_size).to(device)

        hard_update(self.f_target, self.f_critic)

        self.actor = ContinuousGaussianPolicy(self.num_inputs, self.n_preferences, self.action_dim, args.hidden_size).to(self.device)
        self.actor_optim = Adam(self.actor.parameters())

        return None

    def select_action(self, state, preference):
        state = FloatTensor(state).to(self.device).unsqueeze(0)
        preference = FloatTensor(preference).to(self.device).unsqueeze(0)

        action, _ = self.actor.sample(state, preference)

        return action.detach().cpu().numpy()[0]
    
    def act(self, state, preference):
        state = FloatTensor(state).to(self.device).unsqueeze(0)
        preference = FloatTensor(preference).to(self.device).unsqueeze(0)

        action, _ = self.actor.sample(state, preference)

        return action.detach().cpu().numpy()[0]

    def divergence(self, log_pi, prior):
        if self.args.divergence == "alpha":
            if (self.args.alpha != 1) and (self.args.alpha != 0):
                alpha = self.args.alpha
                t = (log_pi.exp()+1e-10)/(prior+1e-10)
                return t.pow(alpha-1)
            elif self.args.alpha == 1:
                return log_pi - (np.log(prior))
            elif self.args.alpha == 0:
                return -prior*torch.log((log_pi.exp()+1e-10)/(prior+1e-10))
        else:
            raise TypeError("Divergence not recognised.")

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        self.actor.train()
        self.f_critic.train()
        self.critic.train()

        state_batch, preference_batch, action_batch, reward_batch, next_state_batch, next_preference_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = FloatTensor(state_batch).repeat(self.n_weights,1).to(self.device)
        next_state_batch = FloatTensor(next_state_batch).repeat(self.n_weights,1).to(self.device)
        action_batch = LongTensor(action_batch).repeat(self.n_weights,1).to(self.device)
        reward_batch = FloatTensor(reward_batch).repeat(self.n_weights,1).to(self.device)
        mask_batch = FloatTensor(mask_batch).repeat(self.n_weights).to(self.device).unsqueeze(1)

        preference_batch = FloatTensor(preference_batch).to(self.device)
        next_preference_batch = FloatTensor(next_preference_batch).to(self.device)

        pref = self.rng.rand(preference_batch.shape[0]*(self.n_weights-1), self.args.num_preferences)
        pref = FloatTensor(pref/np.sum(pref))

        preference_batch = torch.cat((preference_batch, pref), dim=0)
        next_preference_batch = torch.cat((next_preference_batch, pref), dim=0)

        with torch.no_grad():
            reward = torch.sum(preference_batch * reward_batch, dim=-1).reshape(-1,1)
            F_next_target = self.f_target(next_state_batch, next_preference_batch)
            target_G_value = reward + mask_batch * self.gamma * (F_next_target)

        # Two Q-functions to mitigate positive bias in the policy improvement step
        G1, G2 = self.critic(state_batch, preference_batch, action_batch)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        G1_loss = F.mse_loss(G1, target_G_value)
        G2_loss = F.mse_loss(G2, target_G_value)

        G_loss = G1_loss + G2_loss

        # Critic Backwards Step
        self.critic_optim.zero_grad()
        G_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)
        self.critic_optim.step()

        action, log_pi = self.actor.sample(state_batch, preference_batch)

        G1_action0, G2_action0 = self.critic_target(state_batch, preference_batch, action)
        G_action0 = torch.min(G1_action0, G2_action0)

        policy_loss = self.divergence(log_pi, G_action0)
        policy_loss = policy_loss.mean()

        if torch.isnan(policy_loss).any():
            print(policy_loss)
            print(G_action0, G1_action0, G2_action0)
            print(log_pi)

        # clamp policy loss
        policy_loss = policy_loss.clamp(-100, 100)

        # Actor backwards step
        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        self.actor_optim.step()

        F_val = self.f_critic(state_batch, preference_batch)
        # target_F_value = next_G_value - self.alpha*divergance_loss.clamp(-1, 1)
        target_F_value = target_G_value 
        F_loss = F.mse_loss(F_val, target_F_value.detach())   

        # F critic backwards step
        self.f_optim.zero_grad()
        F_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.f_critic.parameters(), max_norm=1)
        self.f_optim.step()     

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.f_target, self.f_critic, self.tau)

        return G_loss.item(), F_loss.item(), policy_loss.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):

        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        
        if not os.path.exists(f"checkpoints/{env_name.replace('-', '_')}"):
            os.makedirs(f"checkpoints/{env_name.replace('-', '_')}")
            
        if ckpt_path is None:
            ckpt_path = "checkpoints/{}/{}_{}".format(env_name.replace('-', '_'), env_name.replace('-', '_'), suffix)
        else:
            ckpt_path = f"checkpoints/{env_name.replace('-', '_')}/" + ckpt_path
        
        print('Saving models to {}'.format(ckpt_path))

        torch.save({'policy_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'f_critic': self.f_critic.state_dict(),
                    'f_critic_optimizer_state_dict': self.f_optim.state_dict(),
                    'f_target': self.f_target.state_dict(),
                    'policy_optimizer_state_dict': self.actor_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):

        print('Loading models from {}'.format(ckpt_path))

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])

            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            self.f_critic.load_state_dict(checkpoint['f_critic'])
            self.f_optim.load_state_dict(checkpoint['f_critic_optimizer_state_dict'])

            self.f_target.load_state_dict(checkpoint["f_target"])

            self.actor_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            self.i_episode = int(ckpt_path.split("_")[-1])

            if evaluate:
                self.actor.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.f_critic.eval()
            else:
                self.actor.train()
                self.critic.train()
                self.critic_target.train()
                self.f_critic.train()

    pass
