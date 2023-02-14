from __future__ import absolute_import, division, print_function
import argparse
import visdom
import torch
import numpy as np
from sklearn.manifold import TSNE

import gym
import sys
import os
import pandas as pd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.mo_env import MultiObjectiveEnv
import mo_gym

speed_param = 0

parser = argparse.ArgumentParser(description='MORL-PLOT')
# CONFIG
parser.add_argument('--env-name', default='ft', metavar='ENVNAME',
                    help='environment to train on (default: tf): ft | ft5 | ft7')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.995, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# PLOT
parser.add_argument('--pltmap', default=False, action='store_true',
                    help='plot deep sea treasure map')
parser.add_argument('--pltpareto', default=True, action='store_true',
                    help='plot pareto frontier')
parser.add_argument('--pltcontrol', default=True, action='store_true',
                    help='plot control curve')
parser.add_argument('--pltdemo', default=False, action='store_true',
                    help='plot demo')
# LOG & SAVING
parser.add_argument('--alpha', type=float, default=5000, metavar='ALPHA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='address for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
# Useless but I am too laze to delete them
parser.add_argument('--mem-size', type=int, default=10000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0., metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=100, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=32, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')

args = parser.parse_args()

args.max_beta = 10
args.g_learning_beta_scheduler = 3000

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

model_list = {
100 : './checkpoint/mo-mountaincar-v0_alpha_-0.5_100',
# 1499 : '../saved_models/multihead2_Q_log_ft_2022_10_13_eps_1499.pkl',
# # 1499 : '../saved_models/multihead3_Q_log_ft_2022_06_22_eps_1499.pkl',
# # 2249 : '../saved_models/multihead3_Q_log_ft_2022_06_22_eps_2249.pkl',
# # 2999 : '../saved_models/multihead3_Q_log_ft_2022_06_22_eps_2999.pkl'
}

env = mo_gym.make('mo-mountaincar-v0')

# generate an agent for plotting
agent = None


from crl.envelope.meta_g_learning_with_oversample import MetaAgent
from crl.envelope.models.inv_rl_r_model import RModel

# import pandas as pd


vis = visdom.Visdom()
assert vis.check_connection()

model_loc = model_list[749]
# model = torch.load(model_loc)

model = torch.load(model_loc).eval()
rmodel = RModel(3, 2, 3).eval()
agent = MetaAgent(model, rmodel, None, args, is_train=False)


opt_x = []
opt_y = []
q_x = []
q_y = []
act_x = []
act_y = []

for i in range(500):
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
            action = agent.act(state, torch.from_numpy(w).type(FloatTensor), FloatTensor([speed_param]).reshape(1,-1))
            next_state, reward, terminal, truncated, info = env.step(action)
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


layout_opt = dict(title="Mountain Car: Q-learning GoD Recovered CCS",
    xaxis=dict(title='Time penalty'),
    yaxis=dict(title='Reverse Penalty'))

vis._send({'data': [act_opt], 'layout': layout_opt})

df = pd.DataFrame(columns=['Train Episode', 'Steps to terminate', 'Reward', 'Left Action', 'Right Action', 'No Action', 'Time Penalty', 'Left penalty', 'Right penalty'])

# for key in model_list.keys():
#     model_loc = model_list[key]
#     model = torch.load(model_loc)
#     agent = MetaAgent(model, None, args, is_train=False)

# probe_list = np.array([[0.9, 0.05, 0.05], [0.9, 0.1, 0.0], [0.9, 0, 0.1], [0.5, 0., 0.5], [0.5, 0.5, 0 ], [0., 0.5, 0.5], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

# for probe in probe_list:
#     steps_to_terminate = []
#     action_count_tracker = {
#         0: [],
#         1: [],
#         2: []
#     }
#     reward_list = []
#     for i in range(100):
#         w_e = probe
#         w = w_e

#         action_count= {
#             0: 0,
#             1: 0,
#             2: 0
#         }


#         env.reset()
#         cnt = 0
#         state = env.reset()[0]
#         terminal = False
#         ttrw = np.array([0.0, 0.0, 0.0])
#         while not terminal:
#             action = agent.act(state, torch.from_numpy(w).type(FloatTensor), FloatTensor([speed_param]).reshape(1,-1))
#             next_state, reward, terminal, truncated, info = env.step(action)
#             state = next_state
#             next_preference = FloatTensor(probe)

#             if next_state[0] - state[0] > 0 and action == 2: 
#                 reward += 0.5
#             if next_state[0] - state[0] < 0 and action == 0: 
#                 reward += 0.5

#             action_count[action] += 1
            
#             if cnt > 300:
#                 terminal = True
#             ttrw = ttrw + reward * np.power(args.gamma, cnt)
#             cnt += 1
        
#         ttrw_w = w.dot(FloatTensor(ttrw))

#         steps_to_terminate.append(cnt)
#         reward_list.append(ttrw_w)
#         action_count_tracker[0].append(action_count[0])
#         action_count_tracker[1].append(action_count[1])
#         action_count_tracker[2].append(action_count[2])
    
#     print('Reward:', np.mean(np.array(reward_list)))
#     print ('steps_to_terminate:' , np.mean(np.array(steps_to_terminate)))
#     print ('Left acceleration mean:',np.mean(np.array(action_count_tracker[0])))
#     print ('Do not accelerate:',np.mean(np.array(action_count_tracker[1])))
#     print ('Right acceleratio:',np.mean(np.array(action_count_tracker[2])))
#     print ('Time Penalty, Left acceleration penalty, Right acceleration penalty:', probe)

#     data = [449, np.mean(np.array(steps_to_terminate)), np.mean(np.array(reward_list)), np.mean(np.array(action_count_tracker[0])), np.mean(np.array(action_count_tracker[2])) ,np.mean(np.array(action_count_tracker[1])), probe[0], probe[1], probe[2]]
    
#     data = np.array(data).reshape(1, -1)
#     df_temp = pd.DataFrame(data, columns=['Train Episode', 'Steps to terminate', 'Reward', 'Left Action', 'Right Action', 'No Action', 'Time Penalty', 'Left penalty', 'Right penalty'])
    
#     df = pd.concat([df, df_temp])      
#     print('-----------------------------------------------------------------')
# df.to_csv('experiment_mt_1.csv')
