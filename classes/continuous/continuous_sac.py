import os
import numpy as np

import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from utils.base_utils import hard_update, soft_update
from .base_classes import Continuous_G_Network, Continuous_F_Network, ContinuousGaussianPolicy
from ..parameters import ContinuousParameters

LOG_SIG_MAX = ContinuousParameters.get_log_sig_max()
LOG_SIG_MIN = ContinuousParameters.get_log_sig_min()
VALUE_SCALING = ContinuousParameters.get_value_scaling()
epsilon = ContinuousParameters.get_epsilon()

torch.autograd.set_detect_anomaly(True)

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
        
        device = ""
        if args.cuda:
            device = "cuda"
        elif args.mps:
            device = "mps"
        else:
            device = "cpu"

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

    # Are the diveregence function outputs the right sign?
    # Add torch.abs() to ensure this so that the divergence is pushed towards zero
    def divergence(self, log_pi: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        if self.args.divergence == "alpha":
            if self.args.alpha != 1 and self.args.alpha != 0:
                alpha = self.args.alpha
                t = (log_pi.exp()+1e-10)/(prior+1e-10)
                return torch.abs(t.pow(alpha-1))
            elif self.args.alpha == 1:
                return torch.abs(log_pi - torch.log(prior))
            elif self.args.alpha == 0:
                return torch.abs(-prior * torch.log((log_pi.exp() + 1e-10) / (prior + 1e-10)))
        else:
            raise ValueError("Unrecognized divergence type.")

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
