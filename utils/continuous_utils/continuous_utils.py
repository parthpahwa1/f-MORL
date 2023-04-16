import math
import time
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pymoo.indicators.hv import HV
from tqdm import tqdm

from classes.parameters import ContinuousParameters
from utils.base_utils import hard_update, soft_update, logistic, hypervolume, find_in

VALUE_SCALING = ContinuousParameters.get_value_scaling()

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


def continuous_train(agent, env, memory, writer, args):
    """
    Train a continuous reinforcement learning agent using the provided environment and hyperparameters.

    Args:
    - agent (object): The reinforcement learning agent to use
    - env (object): The environment to train the agent in
    - memory (object): The memory buffer to store experiences in
    - writer (object): The tensorboard writer object to log metrics to
    - args (object): The command-line arguments and hyperparameters

    Returns:
    - None
    """
    rng = np.random.RandomState(args.seed)
    # Generate a matrix of preferences to be used for evaluation
    pref_list = rng.rand(2000, args.num_preferences)
    pref_list = pref_list / np.sum(pref_list, axis=1)[:, None]
    # pref_list = FloatTensor(pref_list)

    max_steps = args.max_steps

    total_numsteps = 0
    updates = 0

    # Set the lower bound for the episode count based on whether the agent has been trained before
    if agent.i_episode != 0:
        lower_bound = agent.i_episode + 1
    else:
        lower_bound = 0

    # Initialize the preference vector for the current episode
    pref = rng.dirichlet(np.ones(args.num_preferences))

    # Loop over the specified number of episodes
    for i_episode in tqdm(range(lower_bound, args.num_episodes)):
        episode_reward = 0
        episode_steps = 0

        done = False
        # Reset the environment at the beginning of each episode
        state, _ = env.reset()

        # Loop until the episode is complete or the maximum number of steps is reached
        while not done and episode_steps < max_steps:
            # Select an action using the current policy and preference vector
            action = agent.select_action(state, pref)

            # If the number of steps is divisible by the batch size, perform a parameter update
            if (len(memory) > args.batch_size) and (i_episode != 0):
                # Perform multiple updates per step in the environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)
                    # Log the loss values to tensorboard
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    updates += 1

            # Take a step in the environment
            next_state, reward, done, truncated, info = env.step(action)
            episode_steps += 1
            total_numsteps += 1

            # Scale the reward using a logistic function
            reward = logistic(reward, scaling=VALUE_SCALING)

            # Update episode reward with dot product of current preference and reward
            episode_reward += pref.dot(reward).item()

            # Set mask to 1 if episode is not done, otherwise 0
            mask = 1 if not done else 0

            # Generate next preference using Dirichlet distribution
            next_pref = rng.dirichlet(np.ones(args.num_preferences))

            # Append current transition to memory buffer
            memory.push(state, pref, action, reward, next_state, pref, mask, agent)

            # Update current state and preference to the next state and preference
            state = next_state
            pref = next_pref

            writer.add_scalar('Episode Length', episode_steps, i_episode)

        if ((i_episode % 250 == 0) and (i_episode != 0)):

            # Print message to mark the start of evaluation
            print("Starting Evaluation")
            tock = time.perf_counter()

            # Initialize variables for tracking evaluation metrics
            avg_reward = 0.
            eval_reward = []
            reward_list = []

            # Loop through all possible preferences
            for eval_pref in pref_list:
                # Reset environment to starting state
                state, _ = env.reset()
                done = False

                # Initialize variables for tracking episode rewards
                temp_reward = np.zeros(args.num_preferences)
                count = 0

                # Loop through each step of the episode
                while not done:
                    # Take action using current policy and evaluation preference
                    action = agent.act(state, eval_pref)
                    next_state, reward, done, _, _ = env.step(action)

                    # Update total reward for this episode
                    temp_reward = temp_reward + reward * np.power(args.gamma, count)

                    # If the episode is done, add the discounted episode reward to the reward list
                    if done:
                        reward_list.append(temp_reward)
                        # Calculate the evaluation reward using the dot product of the evaluation preference and the episode reward
                        eval_reward.append(np.dot(eval_pref, reward))
                    else:
                        pass

                    # Update current state and step count
                    state = next_state
                    count += 1

            # Remove duplicates from reward list and calculate the average evaluation reward
            reward_list = np.array(list(set([tuple(i) for i in reward_list])))
            avg_reward = sum(eval_reward)/len(eval_reward)

            # Calculate the hypervolume using the discounted rewards for each preference
            hyper = hypervolume(args.ref_point, reward_list)

            # Log evaluation metrics to TensorBoard
            writer.add_scalar('Test Average Reward', avg_reward, i_episode)
            writer.add_scalar('Hypervolume', hyper, i_episode)

            # Print message to mark the end of evaluation
            tick = time.perf_counter()
            print(f"Evaluation completed in {round(tick-tock,2)}")

            print("----------------------------------------")
            # Print evaluation metrics to console
            print("----------------------------------------")
            print(
                "\nEpisode Count: {}; \nHypervolume {}; \nAvg. Reward: {}."
                .format(i_episode, 
                        hyper,
                        round(avg_reward, 2),
                        )
                    )
            print("----------------------------------------")

            # Save agent checkpoint every 1000 episodes
            if (i_episode % 1000 == 0):
                agent.save_checkpoint(args.env_name.replace("-", "_"), ckpt_path=f"{args.env_name.replace('-', '_')}_{args.divergence}_{args.alpha}_{i_episode}")

        # If maximum number of steps or episodes has been reached, end training and print message to console
        if total_numsteps > args.num_steps or i_episode >= args.num_episodes:
            print("Training Complete")
            return None

        env.close()

def continuous_evaluate(agent, env, args):

    rng = np.random.RandomState(args.seed)
    pref_list = rng.rand(2000, args.num_preferences)
    pref_list = pref_list/np.sum(pref_list, axis=1)[:, None]

    if args.env_name == "minecart-v0":
        pareto_df = pd.read_csv("minecart_pareto.csv")

    # Mark start of evaluation.
    print("Starting Evaluation")
    tock = time.perf_counter()

    avg_reward = 0.

    eval_reward = []
    reward_list = []

    for eval_pref in pref_list:
        state, _ = env.reset()

        done = False

        temp_reward = np.zeros(args.num_preferences)
        count = 0
        while not done:
            action = agent.act(FloatTensor(state), FloatTensor(eval_pref))
            next_state, reward, done, _, _ = env.step(action)

            temp_reward = temp_reward + reward * np.power(args.gamma, count)  

            if args.env_name == "deep-sea-treasure-v0":
                if count > 25:
                    # print(f"Breaking {temp_pref} evaluation. Count too high.")
                    done = True

            if args.env_name == "mo-mountaincar-v0":
                if count > 199:
                    # print(f"Breaking {temp_pref} evaluation. Count too high.")
                    done = True

            if args.env_name == "mo-lunar-lander-v2":
                if count > 999:
                    # print(f"Breaking {temp_pref} evaluation. Count too high.")
                    done = True

            if args.env_name == "resource-gathering-v0":
                if count > 100:
                    # print(f"Breaking {temp_pref} evaluation. Count too high.")
                    done = True

            if args.env_name == "minecart-v0":
                if count > 999:
                    # print(f"Breaking {temp_pref} evaluation. Count too high.")
                    done = True

            if done:
                reward_list.append(temp_reward)
                eval_reward.append(np.dot(eval_pref, reward))
            else:
                pass
        
            # print(temp_reward)
            state = next_state

            count += 1

    reward_list = np.array(list(set([tuple(i) for i in reward_list])))
    avg_reward = sum(eval_reward)/len(eval_reward)

    # calculate hyper volume using discounted rewards at the end of episodes for each preference
    hyper = hypervolume(args.ref_point, reward_list)

    if args.env_name == "minecart-v0":
        cnt1, cnt2 = find_in(reward_list, pareto_df, 0.0)
        act_precision = cnt1 / len(reward_list)
        print(act_precision)
        act_recall = cnt2 / len(pareto_df)
        act_f1 = 2 * act_precision * act_recall / (act_precision + act_recall)
    
        # Mark end of evaluation
        tick = time.perf_counter()

        print(f"Evaluation complete in {round(tick-tock,2)}s.")
        env.close()

        return (avg_reward, hyper, act_precision, act_recall, act_f1)

    else:
        tick = time.perf_counter()

        print(f"Evaluation complete in {round(tick-tock,2)}s.")
        env.close()

        return {"Step": agent.i_episode,"avg_reward": avg_reward, "Value": hyper}

