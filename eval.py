from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
import numpy as np
from sklearn.manifold import TSNE
from utils import *
from discrete_models import *
from continuous_models import *
from replay_memory import *
from tqdm import tqdm
import gym
import sys
import os
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import mo_gymnasium


speed_param = 0

parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
arser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="mo-mountaincar-v0",
                    help='MOGYM enviroment (default: mo-mountaincar-v0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.01)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=int(1.5e6), metavar='N',
                    help='maximum number of steps (default: 1.5e6)')
parser.add_argument('--num_episodes', type=int, default=3001, metavar='N',
                    help='maximum number of episodes (default: 3000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default= 1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--mps', action="store_true",
                    help='run on mps (default: False)')

parser.add_argument('--evaluate', action="store_true",
                    help="Evaluate or Train")

parser.add_argument('--divergence', type=str, default='alpha',
                    help="Type of divergence constraint")
parser.add_argument('--alpha', type=float, default=1.0, metavar='G',
                    help='alpha divergence constant (default: 1.0)')

args = parser.parse_args()

args.max_beta = 10
args.g_learning_beta_scheduler = 3000

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

model_list = {}
i_max = 0
for i in range(0,60):
    if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
        model_list[i] = f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"
        i_max = i*50
    else:
        pass

# model_list = {
# 100 : './checkpoint/mo-mountaincar-v0_alpha_-0.5_100',
# # 1499 : '../saved_models/multihead2_Q_log_ft_2022_10_13_eps_1499.pkl',
# # # 1499 : '../saved_models/multihead3_Q_log_ft_2022_06_22_eps_1499.pkl',
# # # 2249 : '../saved_models/multihead3_Q_log_ft_2022_06_22_eps_2249.pkl',
# # # 2999 : '../saved_models/multihead3_Q_log_ft_2022_06_22_eps_2999.pkl'
# }


# generate an agent for plotting
agent = None

vis = visdom.Visdom()
assert vis.check_connection()

# model_loc = model_list[:-1]
# model = torch.load(model_loc)
args.env_name == "mo-mountaincar-v0"
env = mo_gymnasium.make("mo-mountaincar-v0")
env.reset()

args.action_dim = 3
args.num_preferences = 3
args.num_weights = 4
args.action_space = env.action_space
args.num_inputs = env.observation_space.shape[0]
agent = DiscreteSAC(args.num_inputs, args)

if i_max != 0:
    agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i_max}")

opt_x = []
opt_y = []
q_x = []
q_y = []
act_x = []
act_y = []

for i in tqdm(range(500)):
    w = np.random.randn(3)
    w[2] = 0
    w = np.abs(w) / np.linalg.norm(w, ord=1)
    w_e = w
    # w = np.random.dirichlet(np.ones(2))
    # w_e = w / np.linalg.norm(w, ord=2)
    
    
    ttrw = np.array([0.0, 0.0, 0.0])
    terminal = False
    state = env.reset()[0]
    cnt = 0
    reward_list = []

    for j in range(50):
        env.reset()
        cnt = 0
        state = env.reset()[0]
        terminal = False
        probe =  w
        ttrw = np.array([0.0, 0.0, 0.0])
        while not terminal:
            action = agent.select_action(state, torch.from_numpy(w).type(FloatTensor))
            # action = agent.act(state, torch.from_numpy(w).type(FloatTensor), FloatTensor([speed_param]).reshape(1,-1))
            next_state, reward, done, truncated, info = env.step(action) # Step
            state = next_state
            next_preference = FloatTensor(w)
            # reward = _['reward']

            if next_state[0] - state[0] > 0 and action == 2: 
                reward += 0.5
            if next_state[0] - state[0] < 0 and action == 0: 
                reward += 0.5
            
            if cnt > 300:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1
        
        ttrw_w = w.dot(FloatTensor(ttrw))*w_e

        reward_list.append(np.array(ttrw_w))

    ttrw_w = np.mean(np.array(reward_list), axis=0)

    # q_x.append(qc[0].tolist())
    # q_y.append(qc[1].tolist())
    act_x.append(ttrw_w[0].tolist())
    act_y.append(ttrw_w[1].tolist())


act_opt = dict(x=act_x,
                y=act_y,
                mode="markers",
                type='custom',
                marker=dict(
                    symbol="circle",
                    size=1),
                name='policy')

q_opt = dict(x=q_x,
                y=q_y,
                mode="markers",
                type='custom',
                marker=dict(
                    symbol="circle",
                    size=1),
                name='predicted')


layout_opt = dict(title="Mountain Car: Recovered CCS",
    xaxis=dict(title='Time penalty'),
    yaxis=dict(title='Reverse Penalty'))

vis._send({'data': [act_opt], 'layout': layout_opt})

df = pd.DataFrame(columns=['Train Episode', 'Steps to terminate', 'Reward', 'Left Action', 'Right Action', 'No Action', 'Time Penalty', 'Left penalty', 'Right penalty'])

# for key in model_list.keys():
#     model_loc = model_list[key]
#     agent.load_checkpoint(model_loc)

probe_list = np.array([[0.9, 0.05, 0.05], [0.9, 0.1, 0.0], [0.9, 0, 0.1], [0.5, 0., 0.5], [0.5, 0.5, 0 ], [0., 0.5, 0.5], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

for probe in probe_list:
    steps_to_terminate = []
    action_count_tracker = {
        0: [],
        1: [],
        2: []
    }
    reward_list = []
    for i in range(100):
        w_e = probe
        w = w_e

        action_count= {
            0: 0,
            1: 0,
            2: 0
        }


        env.reset()
        cnt = 0
        state = env.reset()[0]
        terminal = False
        ttrw = np.array([0.0, 0.0, 0.0])
        while not terminal:
            action = agent.act(state, torch.from_numpy(w).type(FloatTensor), FloatTensor([speed_param]).reshape(1,-1))
            next_state, reward, terminal, truncated, info = env.step(action)
            state = next_state
            next_preference = FloatTensor(probe)

            if next_state[0] - state[0] > 0 and action == 2: 
                reward += 0.5
            if next_state[0] - state[0] < 0 and action == 0: 
                reward += 0.5

            action_count[action] += 1
            
            if cnt > 300:
                terminal = True
            ttrw = ttrw + reward * np.power(args.gamma, cnt)
            cnt += 1
        
        ttrw_w = w.dot(FloatTensor(ttrw))

        steps_to_terminate.append(cnt)
        reward_list.append(ttrw_w)
        action_count_tracker[0].append(action_count[0])
        action_count_tracker[1].append(action_count[1])
        action_count_tracker[2].append(action_count[2])
    
    print('Reward:', np.mean(np.array(reward_list)))
    print ('steps_to_terminate:' , np.mean(np.array(steps_to_terminate)))
    print ('Left acceleration mean:',np.mean(np.array(action_count_tracker[0])))
    print ('Do not accelerate:',np.mean(np.array(action_count_tracker[1])))
    print ('Right acceleratio:',np.mean(np.array(action_count_tracker[2])))
    print ('Time Penalty, Left acceleration penalty, Right acceleration penalty:', probe)

    data = [449, np.mean(np.array(steps_to_terminate)), np.mean(np.array(reward_list)), np.mean(np.array(action_count_tracker[0])), np.mean(np.array(action_count_tracker[2])) ,np.mean(np.array(action_count_tracker[1])), probe[0], probe[1], probe[2]]
    
    data = np.array(data).reshape(1, -1)
    df_temp = pd.DataFrame(data, columns=['Train Episode', 'Steps to terminate', 'Reward', 'Left Action', 'Right Action', 'No Action', 'Time Penalty', 'Left penalty', 'Right penalty'])
    
    df = pd.concat([df, df_temp])      
    print('-----------------------------------------------------------------')

df.to_csv(f'{args.alpha}_alpha_mt.csv', index=False)
