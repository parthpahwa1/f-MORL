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

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Discrete_F_Network(nn.Module):
    def __init__(self, num_inputs, num_preferences, action_dim, hidden_dim):
        super(Discrete_F_Network, self).__init__()

        self.linear1 = nn.Linear(num_inputs+num_preferences, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2a = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state, preference):
        input = torch.cat([state, preference], 1)
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear2a(x))
        x = self.linear3(x)
        return x


class Discrete_G_Network(nn.Module):
    def __init__(self, num_inputs, num_preferences, action_dim, hidden_dim):
        super(Discrete_G_Network, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_preferences, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_a = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_preferences, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear5_a = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state, preference):
        xu = torch.cat([state, preference], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear2_a(x1))
        x1 = self.linear3(x1)
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = F.relu(self.linear5_a(x2))
        x2 = self.linear6(x2)

        return x1, x2


class DiscreteGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_preferences, action_dim, hidden_dim):
        super(DiscreteGaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_preferences, hidden_dim)
        self.linear2a = nn.Linear(hidden_dim, hidden_dim)
        self.linear2b = nn.Linear(hidden_dim, hidden_dim)
        self.linear2c = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)

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


class DiscreteSAC(object):

    def __init__(self, num_inputs, args):
        super().__init__()

        self.num_inputs = num_inputs

        self.args = args

        self.action_space = args.action_space
        self.num_actions = args.action_space.n
        self.action_dim = args.action_dim
        self.n_preferences = args.num_preferences
        self.n_weights = args.num_weights

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_type = args.policy
        self.alpha = args.alpha
        

        device = ""
        if args.cuda:
            device = "cuda"
        elif args.mps:
            device = "mps"
        else:
            device = "cpu"

        self.device = torch.device(device)

        self.critic = Discrete_G_Network(self.num_inputs, self.n_preferences, self.action_dim, args.hidden_size).to(device)
        self.critic_optim = Adam(self.critic.parameters())

        self.critic_target = Discrete_G_Network(self.num_inputs, self.n_preferences, self.action_dim, args.hidden_size).to(device)
        hard_update(self.critic_target, self.critic)

        self.f_critic = Discrete_F_Network(self.num_inputs, self.n_preferences, self.action_dim, args.hidden_size).to(device)
        self.f_optim = Adam(self.f_critic.parameters())

        self.actor = DiscreteGaussianPolicy(self.num_inputs, self.n_preferences, self.action_dim, args.hidden_size).to(self.device)
        self.actor_optim = Adam(self.actor.parameters())

        return None
    
    def select_action(self, state, preference, evaluate=False):
        state = torch.Tensor(state).to(self.device).unsqueeze(0)
        preference = torch.FloatTensor(preference).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.actor.sample(state, preference)
        else:
            _, _, action = self.actor.sample(state, preference)
        return action.detach().cpu().numpy()[0]
    
    def divergance(self, pi, prior):
        if self.args.divergence == "alpha":
            if (self.args.alpha != 1) and (self.args.alpha != 0):
                alpha = self.args.alpha
                t = (pi+1e-10)/(prior+1e-10)
                return (t.pow(alpha) - alpha*t - (1-alpha))/(alpha*(alpha-1))
            elif self.args.alpha == 1:
                return pi*torch.log((pi+1e-10)/(prior+1e-10))
            elif self.args.alpha == 0:
                return -torch.log((pi+1e-10)/(prior+1e-10))
       
        elif self.args.divergence == "variational_distance":
            return 0.5*torch.abs((pi+1e-10)/(prior+1e-10)-1)
       
        elif self.args.divergence == "Jensen-Shannon":
            t = (pi+1e-10)/(prior+1e-10)
            return -(t+1)*torch.log((t+1)/(2))+t*torch.log(t)
        else:
            pass
         
    def generate_neighbours(self, preference, next_preference, weight_num, speed=None):
        if speed is None:
            speed = torch.zeros((preference.shape[0] * (weight_num-1), 1))

        preference_neigbor = preference.clone().detach()
        next_preference_neigbor = next_preference.clone().detach()

        new_preference = np.random.randn(preference.shape[0] * (weight_num-1), preference.shape[1])
        new_preference = np.abs(new_preference)/np.linalg.norm(new_preference, ord=1,axis=1)[:,None]

        cov = np.identity(new_preference.shape[1])*0.000001

        new_next_preference = np.array([np.random.multivariate_normal(tensor, cov*cov_speed.reshape(-1)[0].item(), 1)[0] for tensor, cov_speed in zip(new_preference, speed)])

        new_next_preference[new_next_preference < 0] = 0
        new_next_preference += 1e-9
        new_next_preference = new_next_preference/np.sum(new_next_preference, axis=1)[:,None]

        preference_neigbor = torch.cat((preference_neigbor, torch.FloatTensor(new_preference)), dim=0)
        next_preference_neigbor = torch.cat((next_preference_neigbor, torch.FloatTensor(new_next_preference)), dim=0)

        return preference_neigbor, next_preference_neigbor

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, preference_batch, action_batch, reward_batch, next_state_batch, next_preference_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device).reshape(-1, 1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        preference_batch = torch.FloatTensor(preference_batch).to(self.device)
        next_preference_batch = torch.FloatTensor(next_preference_batch).to(self.device)
        
        with torch.no_grad():
            reward = torch.sum(preference_batch * reward_batch, dim=-1).reshape(-1,1)
            F_next_target = self.f_critic(next_state_batch, next_preference_batch)
            next_G_value = reward + mask_batch * self.gamma * (F_next_target)
        
        # Two Q-functions to mitigate positive bias in the policy improvement step
        G1, G2 = self.critic(state_batch, preference_batch)  
        
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        G1_loss = F.smooth_l1_loss(G1, next_G_value)  
        G2_loss = F.smooth_l1_loss(G2, next_G_value)  
        
        G_loss = G1_loss + G2_loss

        # Critic Backwards Step
        self.critic_optim.zero_grad()
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optim.step()

        pi = self.actor.get_probs(state_batch, preference_batch)

        G1_action0, G2_action0 = self.critic(state_batch, preference_batch)

        G_action0 = torch.min(G1_action0, G2_action0)

        prior = torch.softmax(G_action0, dim=1)

        divergance_loss = self.divergance(pi, prior.detach())
        divergance_loss = torch.sum(divergance_loss, dim=1).reshape(-1,1)
        policy_loss = divergance_loss.mean()

        # Actor backwards step
        self.actor_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optim.step()

        F_val = self.f_critic(state_batch, preference_batch)
        target_F_value = next_G_value - self.alpha*divergance_loss.clamp(-1, 1)
        F_loss = F.mse_loss(F_val, target_F_value.detach())   

        # F critic backwards step
        self.f_optim.zero_grad()
        F_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.f_critic.parameters(), 1)
        self.f_optim.step()     

            
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (self.log_pi + self.target_entropy).detach()).mean()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
            
        return G1_loss.item(), G_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
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
            self.actor_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.actor.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.actor.train()
                self.critic.train()
                self.critic_target.train()

    pass
