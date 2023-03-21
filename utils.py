import math
import torch
import itertools
import numpy as np
import pandas as pd
import time 
from tqdm import tqdm
from pymoo.indicators.hv import HV

from typing import List, Optional, Tuple, Union

if  torch.cuda.is_available():
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


def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point.
    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors
    Returns:
        float: Hypervolume metric
    """
    ind = HV(ref_point=ref_point*-1)
    return ind(np.array(points)*-1)


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def generate_next_preference(preference, alpha=0.2):
    preference = np.array(preference)
    preference += 1e-6
    
    return FloatTensor(np.random.dirichlet(alpha*preference))


def generate_next_preference_gaussian(preference, alpha=0.2):
    
    cov = np.identity(preference.shape[0])*0.000001*alpha
    new_next_preference = np.random.multivariate_normal(preference, cov, 1)[0]
    new_next_preference[new_next_preference < 0] = 0
    new_next_preference += 1e-9
    new_next_preference = new_next_preference/np.sum(new_next_preference)
    
    return FloatTensor(new_next_preference)


def find_in(A, B, eps=0.2):
    # find element of A in B with a tolerance of relative err of eps.
    cnt1, cnt2 = 0.0, 0.0
    for a in A:
        for b in B:
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt1 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt1 += 1.0
                  break
    for b in B:
        for a in A:
            if eps > 0.001:
              if np.linalg.norm(a - b, ord=1) < eps*np.linalg.norm(b):
                  cnt2 += 1.0
                  break
            else:
              if np.linalg.norm(a - b, ord=1) < 0.5:
                  cnt2 += 1.0
                  break
    return cnt1, cnt2


def discrete_train(agent, env, memory, writer, args):
    rng = np.random.RandomState(args.seed)
    pref_list = rng.rand(2000, args.num_preferences)
    pref_list = pref_list/np.sum(pref_list, axis=1)[:, None]
    pref_list = torch.FloatTensor(pref_list)
    

    if args.env_name == "minecart-v0":
        pareto_df = pd.read_csv("minecart_pareto.csv")
        max_steps = 1000
    elif args.env_name == "mo-minecart-v0":
        pareto_df = pd.read_csv("minecart_pareto.csv")
        max_steps = 200
    else:
        max_steps = 500

    total_numsteps = 0
    updates = 0

    if agent.i_episode != 0:
        lower_bound = agent.i_episode + 1
    else:
        lower_bound = 0

    for i_episode in tqdm(range(lower_bound, args.num_episodes)):
        episode_reward = 0
        episode_steps = 0

        done = False
        state, _ = env.reset()

        if i_episode % 2 == 0:
            pref = rng.dirichlet(np.ones(args.num_preferences))
            pref = torch.FloatTensor(pref)
        else:
            pref = rng.rand(args.num_preferences)
            pref = torch.FloatTensor(pref/np.sum(pref))

        while not done and episode_steps < max_steps:
            action = agent.select_action(state, pref)  # Sample action from policy

            if (total_numsteps+1)%2==0:
                action = rng.randint(0, args.action_space.n)

            # If the number of steps is divisible by the batch size perform an update
            if (len(memory) > args.batch_size) and (i_episode != 0):
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)
                    # Update parameters of all the networks
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', args.alpha, updates)
                    updates += 1
            
            next_state, reward, done, truncated, info = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1

            if args.env_name == "mo-mountaincar-v0":
                if next_state[0] - state[0] > 0 and action == 2: 
                    reward += 0.5
                if next_state[0] - state[0] < 0 and action == 0: 
                    reward += 0.5

            episode_reward += pref.dot(FloatTensor(reward)).item()

            mask = 1 if not done else 0

            memory.push(state, pref, action, reward, next_state, pref, mask, agent) # Append transition to memory

            state = next_state

        writer.add_scalar('reward/train', episode_reward, i_episode)

        if (i_episode % 100 == 0) and (i_episode != 0):
            # Mark start of evaluation.
            print("Starting Evaluation")
            tock = time.perf_counter()

            avg_reward = 0.

            eval_reward = []
            reward_list = []

            if args.env_name != "mo-mountaincar-v0":
                for eval_pref in pref_list:
                    state, _ = env.reset()
                    eval_pref = eval_pref.clone().detach().cpu()
                    temp_pref = eval_pref.numpy()
                    # value_f0 = agent.f_critic(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1))
                    # value_g0 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[0.0]])))[0]
                    # value_g1 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[1.0]])))[0]
                    # value_g2 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[2.0]])))[0]
                    # value_g3 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[3.0]])))[0]
                    done = False

                    temp_reward = np.zeros(args.num_preferences)
                    count = 0
                    while not done:
                        action = agent.act(state, eval_pref)
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
                            eval_reward.append(np.dot(temp_pref, reward))
                        else:
                            pass
                        

                        state = next_state

                        count += 1

                reward_list = np.array(list(set([tuple(i) for i in reward_list])))
                avg_reward = sum(eval_reward)/len(eval_reward)

                # calculate hyper volume using discounted rewards at the end of episodes for each preference
                hyper = hypervolume(args.ref_point, reward_list)

                writer.add_scalar('Test Average Reward', avg_reward, i_episode)
                writer.add_scalar('Hypervolume', hyper, i_episode)

                if args.env_name == "minecart-v0":
                    cnt1, cnt2 = find_in(reward_list, pareto_df.values, 0.001)
                    act_precision = cnt1 / len(reward_list)
                    act_recall = cnt2 / len(pareto_df)

                    if act_precision == 0 and act_recall == 0:
                        act_f1 = 0.0
                    else:
                        act_f1 = 2 * act_precision * act_recall / (act_precision + act_recall)

                    writer.add_scalar('pareto_precision', act_precision, i_episode)
                    writer.add_scalar('pareto_recall', act_recall, i_episode)
                    writer.add_scalar('pareto_f1', act_f1, i_episode)
                
                # Mark end of evaluation
                tick = time.perf_counter()
                print(f"Evaluation completed in {round(tick-tock,2)}")
                
                print(
                        "----------------------------------------"
                        )

                # , Value S0: {}, Value G0: {}, Value G1: {}
                print(
                    "\nEpisode Count: {}; \nHypervolume {}; \nAvg. Reward: {}."
                    .format(i_episode, 
                            hyper,
                            round(avg_reward, 2), 
                            # float(value_f0.detach().cpu().numpy()), 
                            # float(value_g0.detach().cpu().numpy()),
                            # float(value_g1.detach().cpu().numpy())
                            )
                        )
                print("----------------------------------------")

            agent.save_checkpoint(args.env_name, ckpt_path=f"{args.env_name}_{args.divergence}_{args.alpha}_{i_episode}")

        if total_numsteps > args.num_steps or i_episode >= args.num_episodes:
            print("Training Complete")
            return None

        env.close()


def discrete_evaluate(agent, env, args):

    rng = np.random.RandomState(args.seed)
    pref_list = rng.rand(2000, args.num_preferences)
    pref_list = pref_list/np.sum(pref_list, axis=1)[:, None]
    pref_list = torch.FloatTensor(pref_list)

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
        eval_pref = eval_pref.clone().detach().cpu()
        temp_pref = eval_pref.numpy()
        # value_f0 = agent.f_critic(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1))
        # value_g0 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[0.0]])))[0]
        # value_g1 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[1.0]])))[0]
        # value_g2 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[2.0]])))[0]
        # value_g3 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[3.0]])))[0]
        done = False

        temp_reward = np.zeros(args.num_preferences)
        count = 0
        while not done:
            action = agent.act(state, eval_pref)
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
                eval_reward.append(np.dot(temp_pref, reward))
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


def continuous_train(agent, env, memory, writer, args):
    rng = np.random.RandomState(args.seed)
    pref_list = rng.rand(2000, args.num_preferences)
    pref_list = pref_list/np.sum(pref_list, axis=1)[:, None]
    pref_list = torch.FloatTensor(pref_list)

    max_steps = args.max_steps

    total_numsteps = 0
    updates = 0

    if agent.i_episode != 0:
        lower_bound = agent.i_episode + 1
    else:
        lower_bound = 0

    for i_episode in range(lower_bound, args.num_episodes):
        episode_reward = 0
        episode_steps = 0

        done = False
        state, _ = env.reset()

        pref = rng.dirichlet(np.ones(args.num_preferences))
        pref = torch.FloatTensor(pref)

        while not done and episode_steps < max_steps:
            action = agent.select_action(state, pref)  # Sample action from policy

            # Clamp actions here 
            if args.env_name in ["mo-hopper-v4", "mo-halfcheetah-v4"]:
                action = np.clip(action, -1, 1)

            # if (total_numsteps+1)%2==0:
            #     action = rng.randint(0,1,size=args.action_dim)

            # If the number of steps is divisible by the batch size perform an update
            if (len(memory) > args.batch_size) and (i_episode != 0):
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size, updates)
                    # Update parameters of all the networks
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    updates += 1
            
            next_state, reward, done, truncated, info = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1

            episode_reward += pref.dot(FloatTensor(reward)).item()

            mask = 1 if not done else 0

            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            reward = torch.FloatTensor(reward)

            memory.push(state, pref, action, reward, next_state, pref, mask, agent) # Append transition to memory

            state = next_state

        writer.add_scalar('reward/train', episode_reward, i_episode)

        if (i_episode % 1000 == 0) and (i_episode != 0):
            # Mark start of evaluation.
            print("Starting Evaluation")
            tock = time.perf_counter()

            avg_reward = 0.

            eval_reward = []
            reward_list = []

            for eval_pref in pref_list:
                state, _ = env.reset()
                eval_pref = eval_pref.clone().detach().cpu()
                temp_pref = eval_pref.numpy()
                done = False

                temp_reward = np.zeros(args.num_preferences)
                count = 0
                while not done:
                    action = agent.act(state, eval_pref)
                    next_state, reward, done, _, _ = env.step(action)

                    temp_reward = temp_reward + reward * np.power(args.gamma, count)

                    if done:
                        reward_list.append(temp_reward)
                        eval_reward.append(np.dot(temp_pref, reward))
                    else:
                        pass

                    state = next_state

                    count += 1

            reward_list = np.array(list(set([tuple(i) for i in reward_list])))
            avg_reward = sum(eval_reward)/len(eval_reward)

            # calculate hyper volume using discounted rewards at the end of episodes for each preference
            hyper = hypervolume(args.ref_point, reward_list)

            writer.add_scalar('Test Average Reward', avg_reward, i_episode)
            writer.add_scalar('Hypervolume', hyper, i_episode)

            # Mark end of evaluation
            tick = time.perf_counter()
            print(f"Evaluation completed in {round(tick-tock,2)}")

            print("----------------------------------------")

            print(
                "\nEpisode Count: {}; \nHypervolume {}; \nAvg. Reward: {}."
                .format(i_episode, 
                        hyper,
                        round(avg_reward, 2),
                        )
                    )
            print("----------------------------------------")

            agent.save_checkpoint(args.env_name, ckpt_path=f"{args.env_name}_{args.divergence}_{args.alpha}_{i_episode}")

        if total_numsteps > args.num_steps or i_episode >= args.num_episodes:
            print("Training Complete")
            return None

        env.close()


def continuous_evaluate(agent, env, args):

    rng = np.random.RandomState(args.seed)
    pref_list = rng.rand(2000, args.num_preferences)
    pref_list = pref_list/np.sum(pref_list, axis=1)[:, None]
    pref_list = torch.FloatTensor(pref_list)

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
        eval_pref = eval_pref.clone().detach().cpu()
        temp_pref = eval_pref.numpy()
        # value_f0 = agent.f_critic(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1))
        # value_g0 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[0.0]])))[0]
        # value_g1 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[1.0]])))[0]
        # value_g2 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[2.0]])))[0]
        # value_g3 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_pref.reshape(1,-1), torch.FloatTensor(np.array([[3.0]])))[0]
        done = False

        temp_reward = np.zeros(args.num_preferences)
        count = 0
        while not done:
            action = agent.act(state, eval_pref)
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
                eval_reward.append(np.dot(temp_pref, reward))
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
    