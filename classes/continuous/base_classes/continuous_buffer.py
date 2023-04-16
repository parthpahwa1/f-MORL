import os
import numpy as np
from utils.base_utils import hard_update, soft_update

import random
import numpy as np
import pickle
import os
import torch

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


class ContinuousMemory:
    def __init__(self, capacity, gamma, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.gamma = gamma
        self.priority_mem = []
        self.position = 0

    def push(self, state: FloatTensor, preference: FloatTensor, action: FloatTensor, reward: float,
             next_state: FloatTensor, next_preference: FloatTensor, done: bool, agent) -> None:
        """
        Adds a new experience tuple to the buffer.

        Args:
            state (FloatTensor): The state of the environment at the current time step.
            preference (FloatTensor): The preference vector of the agent at the current time step.
            action (FloatTensor): The action taken by the agent at the current time step.
            reward (float): The reward received by the agent at the current time step.
            next_state (FloatTensor): The state of the environment at the next time step.
            next_preference (FloatTensor): The preference vector of the agent at the next time step.
            done (bool): Whether the episode has ended after the current time step.
            agent: The agent object that interacts with the environment and updates its parameters.

        Returns:
            None
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priority_mem.append(None)

        self.buffer[self.position] = (state, preference, action, reward, next_state, next_preference, done)

        state_input = FloatTensor(state).unsqueeze(0)
        preference_input = FloatTensor(preference).unsqueeze(0)
        action_input = FloatTensor(action).unsqueeze(0)

        cond_one = torch.isnan(state_input).any()
        cond_two = torch.isnan(preference_input).any()
        cond_three = torch.isnan(action_input).any()

        if ((cond_one or cond_two) or (cond_three)):
            print(state, "\n \n", preference, "\n \n", action, "\n \n")

        for param in agent.critic_target.parameters():
            if torch.isnan(param).any():
                print(param)

        agent.critic_target.eval()
        Q_val_0, Q_val_1 = agent.critic_target(state_input, preference_input, action_input)
        agent.critic_target.train()

        Q_val = torch.min(Q_val_0, Q_val_1)

        agent.critic.eval()
        hq_0, hq_1 = agent.critic(state_input, preference_input, action_input)
        agent.critic.train()

        if torch.isnan(hq_0).any() or torch.isnan(hq_1).any():
            print(hq_0, hq_1, "\n \n", Q_val_0, Q_val_1)

        hq = torch.min(hq_0, hq_1)
        hq = torch.max(hq)

        wr = preference.dot(reward)
        p = abs(wr + done*self.gamma * hq - Q_val)

        self.priority_mem[self.position] = p.detach().cpu().numpy()

        self.position = (self.position + 1) % self.capacity

        return None

    def sample(self, batch_size: int) -> tuple:
        pri = np.array(self.priority_mem)
        pri = pri / pri.sum()

        pri = pri.flatten()

        index_list = np.random.choice(
            range(len(self.buffer)), batch_size,
            replace=False,
            p=pri
        )

        batch = [self.buffer[i] for i in index_list]

        state, preference, action, reward, next_state, next_preference, done = map(np.stack, zip(*batch))

        return state, preference, action, reward, next_state, next_preference, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if not os.path.exists(f'checkpoints/{env_name}'):
            os.makedirs(f'checkpoints/{env_name}')

        if save_path is None:
            save_path = "checkpoints/{}/{}_{}".format(env_name,env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

