import os
import datetime
import argparse
import copy
import multiprocessing as mp
from tqdm import tqdm
from typing import Optional
from functools import partial

import numpy as np
import mo_gymnasium
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from classes import ContinuousMemory, ContinuousSAC
from utils import continuous_evaluate, continuous_train

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--env_name', default="mo-hopper-v4",
                    help='MOGYM environment (default: mo-hopper-v4)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=int(1e6), metavar='N',
                    help='maximum number of steps (default: 1e6)')
parser.add_argument('--num_episodes', type=int, default=int(1e6), metavar='N',
                    help='maximum number of episodes (default: 100000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=int(1e6), metavar='N',
                    help='size of replay buffer (default: 1e6)')
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

# Assertions
assert args.divergence in {"alpha"}
assert args.env_name in {"mo-hopper-v4", "mo-halfcheetah-v4"}

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

def multi_train(args_input, env, alpha: float) -> Optional[None]:
    """
    Trains a ContinuousSAC agent and saves the checkpoint and TensorBoard logs.
    :param args_input: The arguments for the training process.
    :param alpha: The learning rate for the training process.
    :return: None.
    """
    # Make a deep copy of the input arguments to avoid modifying the original object.
    args_copy = copy.deepcopy(args_input)
    
    # Set the learning rate of the agent to the given alpha.
    args_copy.alpha = alpha
    
    # Create a ContinuousSAC agent with the given input arguments.
    agent = ContinuousSAC(args_copy.num_inputs, args_copy)

    # Find the latest checkpoint file and load it if it exists.
    i_max = 0
    for i in range(0, 60):
        checkpoint_path = f"checkpoints/{args_copy.env_name.replace('-', '_')}/{args_copy.env_name.replace('-', '_')}_{args_copy.divergence}_{args_copy.alpha}_{i * 50}"
        if os.path.exists(checkpoint_path):
            i_max = i * 50
        else:
            pass

    if i_max != 0:
        agent.load_checkpoint(f"checkpoints/{args_copy.env_name.replace('-', '_')}/{args_copy.env_name.replace('-', '_')}_{args_copy.divergence}_{args_copy.alpha}_{i_max}")

    # If evaluate flag is not set, train the agent and save the TensorBoard logs.
    if not args_copy.evaluate:
        memory = ContinuousMemory(args_copy.replay_size, args_copy.gamma, args_copy.seed)
        log_dir = f"./tensorboard_logs/{args_copy.env_name.replace('-', '_')}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_SAC_{args_copy.env_name.replace('-', '_')}_{args_copy.divergence}_{args_copy.alpha}"
        writer = SummaryWriter(log_dir)
        continuous_train(agent, env, memory, writer, args_copy)
    else:
        # If evaluate flag is set, evaluate the agent on the environment and print the results.
        print(continuous_evaluate(agent, env, args_copy))
    
    return None

if __name__ == "__main__":

    env = mo_gymnasium.make(args.env_name)
    env.reset()

    if args.env_name == "mo-hopper-v4":

        env = mo_gymnasium.make(args.env_name, cost_objective=False)
        env.reset()
        
        args.env_name = args.env_name.replace('-', '_')
        args.action_dim = 3
        args.num_preferences = 2
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.zeros(args.num_preferences)
        args.max_steps = 1000
        
        func = partial(multi_train, args_input=args, env=env)

        func(alpha=args.alpha)

    elif args.env_name == "mo-halfcheetah-v4":

        args.action_dim = 6
        args.num_preferences = 2
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.zeros(args.num_preferences)
        args.max_steps = 1000

        func = partial(multi_train, args_input=args, env=env)

        func(alpha=args.alpha)
     

    else:
        raise NameError(f"{args.env_name} is not an enviroment.")

