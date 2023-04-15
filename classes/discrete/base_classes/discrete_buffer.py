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


class DiscreteMemory:
    def __init__(self, capacity, gamma, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.gamma = gamma
        self.priority_mem = []
        self.position = 0

    def push(self, state, preference, action, reward, next_state, next_preference, done, agent):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priority_mem.append(None)

        self.buffer[self.position] = (state, preference, action, reward, next_state, next_preference, done)
        
        agent.critic_target.eval()
        Q_val_0, Q_val_1 = agent.critic_target(FloatTensor(state.reshape(1,-1)), preference.reshape(1,-1))
        Q_val = torch.min(Q_val_0, Q_val_1)[0]
        Q_val = Q_val[action]

        agent.critic_target.eval()
        hq_0, hq_1 = agent.critic_target(FloatTensor(next_state.reshape(1,-1)), next_preference.reshape(1,-1))
        hq = torch.min(hq_0, hq_1)
        hq = torch.max(hq)

        wr = preference.dot(FloatTensor(reward))
        p = abs(wr + done*self.gamma * hq - Q_val)

        self.priority_mem[self.position] = p.detach().cpu().numpy()

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        pri =  np.array(self.priority_mem)
        pri = pri / pri.sum()

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

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

    pass
