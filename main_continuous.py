import argparse
import datetime
import numpy as np
from utils import *
from continuous_models import *
from replay_memory import *
import mo_gymnasium

import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="mo-hopper-v4",
                    help='MOGYM enviroment (default: mo-hopper-v4)')
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

# Assertions
assert args.divergence in {"alpha", "variational_distance", "Jensen-Shannon"}
assert args.env_name in {"mo-hopper-v4"}

if  torch.cuda.is_available():
    if args.cuda:
        device = torch.device("cuda")
        FloatTensor = torch.cuda.FloatTensor 
        LongTensor = torch.cuda.LongTensor 
        ByteTensor = torch.cuda.ByteTensor 
        Tensor = FloatTensor
else:
    FloatTensor = torch.FloatTensor 
    LongTensor = torch.LongTensor 
    ByteTensor = torch.ByteTensor 
    Tensor = FloatTensor

if __name__ == "__main__":

    env = mo_gymnasium.make(args.env_name)
    env.reset()

    if args.env_name == "mo-hopper-v4":
        args.action_dim = 3
        args.num_preferences = 3
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.zeros(args.num_preferences)
        args.max_steps = 1000

        agent = ContinuousSAC(args.num_inputs, args)

        i_max = 0
        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                i_max = i*50
            else:
                pass

        if i_max != 0:
            agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i_max}")

        if not args.evaluate:
            memory = ContinuousMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/{args.env_name.replace("-", "_")}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            continuous_train(agent, env, memory, writer, args)
        else:
            print(continuous_evaluate(agent, env, args))


    else:
        raise NameError(f"{args.env_name} is not an enviroment.")

