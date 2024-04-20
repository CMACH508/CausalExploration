import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from algorithms.DQN_policy import DQN
from algorithms.CD_alg import select_batch_data, get_structure_pc

from models.Environment import GroundEnv
from models.CausalWorldModel import WorldModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(n_samples, env, agent, worldmodel, enviroment_model):
    print('Enter training iteration...')
    sample_batch = 150
    has_factorize = False
    data_list, next_data_list, reward_list, mean_reward_list = [],[],[],[]

    a_1 = np.random.randint(0, agent.action_range, size=env.action_dim) / (agent.action_range - 1)
    input_obs_action = np.append(env.z_1, a_1).reshape(1, -1)

    for step in range(n_samples + 1):
        next_obs = env(input_obs_action)
        action_value = agent.choose_action(next_obs.reshape(1, -1))

        if step == sample_batch:
            causal_graph_pred = get_structure_pc(np.array(data_list), np.array(next_data_list))
            worldmodel.set_net(causal_graph_pred, enviroment_model)
            has_factorize = True
        if step > sample_batch and step % sample_batch == 0:
            batch_id = select_batch_data(data_list, next_data_list, step, worldmodel, 2 * sample_batch, has_factorize)
            causal_graph_pred = get_structure_pc(np.array(data_list)[batch_id], np.array(next_data_list)[batch_id])

        if step > 0:
            data_list.append(input_obs_action.reshape(-1))
            next_data_list.append(next_obs.reshape(-1))
            reward = worldmodel.get_reward(input_obs_action, next_obs, has_factorize)
            reward_list.append(reward)
            if step < 32:
                mean_reward_list.append(np.mean(reward_list).reshape(-1))
            else:
                mean_reward_list.append(np.mean(reward_list[-32:]).reshape(-1))
            agent.store_transition(input_obs_action.reshape(-1), reward, next_obs)
            worldmodel.store_data(input_obs_action.reshape(-1), next_obs)
        
        input_obs_action = np.append(next_obs, action_value).reshape(1, -1)

        if (step >= 64) and (step % 20 == 0):
            agent.learn()
        if (step >= 64) and (step % 32 == 0):
            worldmodel.train(has_factorize)    
    print('training success!')


def Synthetic_Exp(enviroment_model='linear'):
    nn_environment = GroundEnv(state_dim, action_dim, enviroment_model)
    agent = DQN(state_dim, action_dim, action_range)
    worldmodel = WorldModel(state_dim, action_dim, enviroment_model)
    train(n_samples, nn_environment, agent, worldmodel, enviroment_model)


if __name__ == '__main__':
    state_dim = 5
    action_dim = 3
    action_range = 16
    env_model = 'linear'
    n_samples = 500 if env_model == 'linear' else 1500
    Synthetic_Exp(enviroment_model=env_model)
