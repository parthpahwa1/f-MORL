import argparse
import os
import datetime
import numpy as np
from utils import *
from classes import DiscreteSAC, DiscreteMemory
import mo_gymnasium

import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="fruit-tree-v0",
                    help='MOGYM enviroment (default: fruit-tree-v0)')
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
assert args.env_name in {"fruit-tree-v0", "mo-lunar-lander-v2", "deep-sea-treasure-v0", 
                         "minecart-v0", "four-room-v0", "resource-gathering-v0", "mo-mountaincar-v0"}

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

    if args.env_name == "fruit-tree-v0":
        args.action_dim = 2
        args.num_preferences = 6
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.zeros(args.num_preferences)

        agent = DiscreteSAC(args.num_inputs, args)

        i_max = 0
        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                i_max = i*50
            else:
                pass

        if i_max != 0:
            agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i_max}")

        if not args.evaluate:
            memory = DiscreteMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/FruitTree_v0/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            discrete_train(agent, env, memory, writer, args)
        else:
            print(discrete_evaluate(agent, env, args))

    elif args.env_name == "mo-lunar-lander-v2":
        args.action_dim = 4
        args.num_preferences = 4
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.array([-100, -100, -100, -100])

        agent = DiscreteSAC(args.num_inputs, args)

        i_max = 0
        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                i_max = i*50
            else:
                pass

        if i_max != 0:
            agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i_max}")

        if not args.evaluate:
            memory = DiscreteMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/LunarLander_v2/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            discrete_train(agent, env, memory, writer, args)
        else:
            print(discrete_evaluate(agent, env, args))

    elif args.env_name == "deep-sea-treasure-v0":
        args.action_dim = 4
        args.num_preferences = 2
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.array([0,-19])

        agent = DiscreteSAC(args.num_inputs, args)

        i_max = 0
        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                i_max = i*50
            else:
                pass

        if i_max != 0:
            agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i_max}")

        if not args.evaluate:
            memory = DiscreteMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/DeepSeaTreasure_v0/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            discrete_train(agent, env, memory, writer, args)
        else:
            print(discrete_evaluate(agent, env, args))

    elif args.env_name == "minecart-v0":
        args.action_dim = 6
        args.num_preferences = 3
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.array([-1e-3,-1e-3,-100])
       
        agent = DiscreteSAC(args.num_inputs, args)
        
        i_max = 0
        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                i_max = i*50
            else:
                pass

        if i_max != 0:
            agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i_max}")

        if not args.evaluate:
            memory = DiscreteMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/Minecart_v0/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            discrete_train(agent, env, memory, writer, args)
        else:
            print(discrete_evaluate(agent, env, args))

    elif args.env_name == "mo-mountaincar-v0":
        args.action_dim = 3
        args.num_preferences = 3
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.array([-10,-10,-10])
       
        agent = DiscreteSAC(args.num_inputs, args)
        
        i_max = 0
        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                i_max = i*50
            else:
                pass

        if i_max != 0:
            agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i_max}")

        if not args.evaluate:
            memory = DiscreteMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/MountainCar_v0/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            discrete_train(agent, env, memory, writer, args)
        else:
            args.ref_point = np.array([-199,-10,-10])
            print(discrete_evaluate(agent, env, args))

    elif args.env_name == "four-room-v0":
        args.action_dim = 4
        args.num_preferences = 3
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.array([0,0,0])

        agent = DiscreteSAC(args.num_inputs, args)

        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                agent = DiscreteSAC(args.num_inputs, args)
                agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}")
            else:
                pass

        if not args.evaluate:
            memory = DiscreteMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/FourRoom_v0/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            discrete_train(agent, env, memory, writer, args)
        else:
            print(discrete_evaluate(agent, env, args))

    elif args.env_name == "resource-gathering-v0":
        args.action_dim = 4
        args.num_preferences = 3
        args.num_weights = 4
        args.action_space = env.action_space
        args.num_inputs = env.observation_space.shape[0]
        args.ref_point = np.array([-1e-3,-1e-3,-0.33])

        agent = DiscreteSAC(args.num_inputs, args)

        for i in range(0,60):
            if os.path.exists(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}"):
                agent = DiscreteSAC(args.num_inputs, args)
                agent.load_checkpoint(f"checkpoints/{args.env_name}_{args.divergence}_{args.alpha}_{i*50}")
            else:
                pass

        if not args.evaluate:
            memory = DiscreteMemory(args.replay_size,  args.gamma, args.seed)
            writer = SummaryWriter(f'./tensorboard_logs/ResourceGathering_v0/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_SAC_{args.env_name}_{args.divergence}_{args.alpha}')
            discrete_train(agent, env, memory, writer, args)
        else:
            print(discrete_evaluate(agent, env, args))

    else:
        raise NameError(f"{args.env_name} is not an enviroment.")

