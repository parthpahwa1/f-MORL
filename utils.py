import math
import torch
import itertools
import numpy as np
from pymoo.indicators.hv import HV
from typing import List, Optional, Tuple, Union

from pymoo.config import Config
Config.warnings['not_compiled'] = False


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
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


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


def train_ft(agent, env, memory, writer, args):
    rng = np.random.RandomState(args.seed)
    prob_list = rng.rand(1000, args.num_preferences)
    prob_list = [torch.FloatTensor(item/sum(item)) for item in prob_list]

    total_numsteps = 0
    updates = 0
    for i_episode in range(args.num_episodes):

        episode_reward = 0
        episode_steps = 0

        done = False
        state, _ = env.reset()
        
        probe = np.random.randn(args.num_preferences)
        probe = generate_next_preference(np.abs(probe)/np.linalg.norm(probe, ord=1), alpha=args.alpha)

        while not done and episode_steps < 500 and i_episode < args.num_episodes:
            action = agent.select_action(state, probe, (i_episode+1)%2==0)  # Sample action from policy

            # epsilon
            if (total_numsteps+1)%2==0:
                action = np.random.randint(0, 2)
                        
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, args.alpha = agent.update_parameters(memory, args.batch_size, updates)
                    # Update parameters of all the networks
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', args.alpha, updates)
                    updates += 1
            
            next_state, reward, done, truncated, info = env.step(action) # Step

            episode_steps += 1
            total_numsteps += 1

            episode_reward += probe.dot(FloatTensor(reward)).item()

            mask = 1 if episode_steps == 20 else float(not done)

            memory.push(state, probe, action, reward, next_state, probe, mask, agent) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps or i_episode >= args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 250 == 0 and args.eval is True:
            avg_reward = 0.

            eval_reward = []
            reward_list = []

            for eval_probe in prob_list:
                state, _ = env.reset()
                value_f0 = agent.f_critic(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1))
                value_g0 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1), torch.FloatTensor(np.array([[0.0]])))[0]
                value_g1 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1), torch.FloatTensor(np.array([[1.0]])))[0]
                # episode_reward = 0

                done = False
                while not done:
                    action = agent.select_action(state, eval_probe, evaluate=True)
                    next_state, reward, done, truncated, info = env.step(action)
                    eval_probe = eval_probe.clone().detach().cpu()
                    eval_reward.append(np.dot(eval_probe, reward))
                    
                    reward_list.append(reward)

                    state = next_state

                avg_reward += sum(eval_reward)/len(eval_reward)

            avg_reward = avg_reward/len(prob_list)
            hyper = hypervolume(args.num_preferences, reward_list)

            writer.add_scalar('Test Average Reward', avg_reward, i_episode)
            writer.add_scalar('Hypervolume', hyper, i_episode)

            print(
                    "----------------------------------------"
                    )

            # , Value S0: {}, Value G0: {}, Value G1: {}
            print(
                "Episode Count: {}; \nHyper Volume: {}; \nAvg. Reward: {}."
                .format(i_episode, 
                        round(hyper,2),
                        round(avg_reward, 2), 
                        # float(value_f0.detach().cpu().numpy()), 
                        # float(value_g0.detach().cpu().numpy()),
                        # float(value_g1.detach().cpu().numpy())
                        )
                    )
            print("----------------------------------------")
    
        if i_episode == args.num_episodes:
            print("Training Complete")
            return None

        env.close()


def train_ll(agent, env, memory, writer, args):
    rng = np.random.RandomState(args.seed)
    prob_list = rng.rand(1000, args.num_preferences)
    prob_list = [torch.FloatTensor(item/sum(item)) for item in prob_list]

    total_numsteps = 0
    updates = 0
    for i_episode in range(args.num_episodes):

        episode_reward = 0
        episode_steps = 0

        done = False
        state, _ = env.reset()
        
        probe = np.random.randn(args.num_preferences)
        probe = generate_next_preference(np.abs(probe)/np.linalg.norm(probe, ord=1), alpha=args.alpha)

        while not done and episode_steps < 500 and i_episode < args.num_episodes:
            action = agent.select_action(state, probe, (i_episode+1)%2==0)  # Sample action from policy
            # epsilon
            # if (total_numsteps+1)%2==0:
            #     action = np.random.randint(0, 2)
                        
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, args.alpha = agent.update_parameters(memory, args.batch_size, updates)
                    # Update parameters of all the networks
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', args.alpha, updates)
                    updates += 1
            
            next_state, reward, done, truncated, info = env.step(action) # Step

            episode_steps += 1
            total_numsteps += 1

            episode_reward += probe.dot(FloatTensor(reward)).item()

            mask = 1 if episode_steps == 20 else float(not done)

            memory.push(state, probe, action, reward, next_state, probe, mask, agent) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps or i_episode >= args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 250 == 0 and args.eval is True:
            avg_reward = 0.

            eval_reward = []
            reward_list = []

            for eval_probe in prob_list:
                state, _ = env.reset()
                value_f0 = agent.f_critic(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1))
                value_g0 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1), torch.FloatTensor(np.array([[0.0]])))[0]
                value_g1 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1), torch.FloatTensor(np.array([[1.0]])))[0]
                # episode_reward = 0

                done = False
                while not done:
                    action = agent.select_action(state, eval_probe, evaluate=True)
                    next_state, reward, done, truncated, info = env.step(action)
                    eval_probe = eval_probe.clone().detach().cpu()
                    eval_reward.append(np.dot(eval_probe, reward))
                    
                    reward_list.append(reward)

                    state = next_state

                avg_reward += sum(eval_reward)/len(eval_reward)

            avg_reward = avg_reward/len(prob_list)
            hyper = hypervolume(args.num_preferences, reward_list)

            writer.add_scalar('Test Average Reward', avg_reward, i_episode)
            writer.add_scalar('Hypervolume', hyper, i_episode)

            print(
                    "----------------------------------------"
                    )

            # , Value S0: {}, Value G0: {}, Value G1: {}
            print(
                "Episode Count: {}; \nHyper Volume: {}; \nAvg. Reward: {}."
                .format(i_episode, 
                        round(hyper,2),
                        round(avg_reward, 2), 
                        # float(value_f0.detach().cpu().numpy()), 
                        # float(value_g0.detach().cpu().numpy()),
                        # float(value_g1.detach().cpu().numpy())
                        )
                    )
            print("----------------------------------------")
    
        if i_episode == args.num_episodes:
            print("Training Complete")
            return None

        env.close()


def train_hopper(agent, env, memory, writer, args):
    rng = np.random.RandomState(args.seed)
    prob_list = rng.rand(1000, args.num_preferences)
    prob_list = [torch.FloatTensor(item/sum(item)) for item in prob_list]

    total_numsteps = 0
    updates = 0
    for i_episode in range(args.num_episodes):

        episode_reward = 0
        episode_steps = 0

        done = False
        state, _ = env.reset()
        
        probe = np.random.randn(args.num_preferences)
        probe = generate_next_preference(np.abs(probe)/np.linalg.norm(probe, ord=1), alpha=args.alpha)

        while not done and episode_steps < 500 and i_episode < args.num_episodes:
            action = agent.select_action(state, probe, (i_episode+1)%2==0)  # Sample action from policy
            
            # epsilon
            if (total_numsteps+1)%2==0:
                action = rng.uniform(-1,1,3)  

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, args.alpha = agent.update_parameters(memory, args.batch_size, updates)
                    # Update parameters of all the networks
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', args.alpha, updates)
                    updates += 1
            
            next_state, reward, done, truncated, info = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1

            episode_reward += probe.dot(FloatTensor(reward)).item()

            mask = 1 if episode_steps == 20 else float(not done)

            memory.push(state, probe, action, reward, next_state, probe, mask, agent) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps or i_episode >= args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.

            eval_reward = []
            reward_list = []

            for eval_probe in prob_list:
                state, _ = env.reset()
                value_f0 = agent.f_critic(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1))
                value_g0 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1), torch.FloatTensor(np.array([[-1.0,-1.0,-1.0]])))[0]
                value_g1 = agent.critic_target(torch.FloatTensor(state.reshape(1,-1)), eval_probe.reshape(1,-1), torch.FloatTensor(np.array([[1.0, 1.0, 1.0]])))[0]
                # episode_reward = 0

                done = False
                while not done:
                    action = agent.select_action(state, eval_probe, evaluate=True)
                    next_state, reward, done, truncated, info = env.step(action)
                    eval_probe = eval_probe.clone().detach().cpu()
                    eval_reward.append(np.dot(eval_probe, reward))
                    
                    reward_list.append(reward)

                    state = next_state

                avg_reward += sum(eval_reward)/len(eval_reward)

            avg_reward = avg_reward/len(prob_list)
            hyper = hypervolume(np.zeros(args.num_preferences), reward_list)

            writer.add_scalar('Test Average Reward', avg_reward, i_episode)
            writer.add_scalar('Hypervolume', hyper, i_episode)

            print(
                    "----------------------------------------"
                    )

            # , Value S0: {}, Value G0: {}, Value G1: {}
            print(
                "Episode Count: {}; \nHyper Volume: {}; \nAvg. Reward: {}."
                .format(i_episode, 
                        round(hyper,2),
                        round(avg_reward, 2), 
                        # float(value_f0.detach().cpu().numpy()), 
                        # float(value_g0.detach().cpu().numpy()),
                        # float(value_g1.detach().cpu().numpy())
                        )
                    )
            print("----------------------------------------")
    
        if i_episode == args.num_episodes:
            print("Training Complete")
            return None

        env.close()

