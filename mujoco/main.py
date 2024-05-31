from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os

from agents.ppo import PPO
from models.causal_world import Causal_world
from causal_discovery.CD_alg import get_structure_pc
from utils.utils import make_transition, Dict, RunningMeanStd


parser = ArgumentParser('parameters')
parser.add_argument("--env_name", type=str, default = 'Reacher-v2')
args = parser.parse_args()

env = gym.make(args.env_name)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)

parser = ConfigParser()
parser.read('config.ini')
args = Dict(parser, 'ppo')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
        

def train(agent, collect_data=True):
    state_ = (env.reset())
    state = np.clip((state_[0] - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)

    for n_epi in range(args.epochs):
        for t in range(args.traj_length):
            if args.render:    
                env.render()
            mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
            dist = torch.distributions.Normal(mu,sigma[0])
            action = dist.sample()

            log_prob = dist.log_prob(action).sum(-1,keepdim = True)
            next_state_, reward, done, info, _ = env.step(action.cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)

            state_action = np.hstack((state, action.cpu().numpy()))
            if collect_data:                    
                transition = make_transition(
                    state,action.cpu().numpy(),
                    np.array([reward*args.exreward_scaling]),
                    next_state,
                    np.array([done]),
                    log_prob.detach().cpu().numpy()
                )                                      
            else:
                pred_error = world_model.get_error(torch.from_numpy(state_action).float().to(device), torch.from_numpy((next_state)).float().to(device))    
                transition = make_transition(
                    state,
                    action.cpu().numpy(),
                    np.array([reward*args.exreward_scaling + pred_error*args.inreward_scaling]),
                    next_state,
                    np.array([done]),
                    log_prob.detach().cpu().numpy()
                )

            agent.put_data(transition) 
            if done:
                state_ = (env.reset())
                state = np.clip((state_[0] - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            else:
                state = next_state
                state_ = next_state_
        
        if not collect_data:
            agent.train_net(n_epi)
            world_model.train_model(n_epi)            
            
    
if __name__ == '__main__':
    for _ in range(args.num_exp):
        rollouts = PPO(args.writer, device, state_dim, action_dim, args)
        agent = PPO(args.writer, device, state_dim, action_dim, args)

        if (torch.cuda.is_available()) and (args.use_cuda):
            rollouts = rollouts.cuda()
            agent = agent.cuda()               

        train(rollouts)        
        data = rollouts.data.sample(shuffle=False)        
        causal_graph = get_structure_pc(np.hstack((data['state'], data['action'])), data['next_state'])

        world_model = Causal_world(args.writer, device, state_dim, action_dim, args, causal_graph)
        if (torch.cuda.is_available()) and (args.use_cuda):
            world_model = world_model.cuda()                        
        train(agent, collect_data=False)
