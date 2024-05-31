from networks.world_net import Causal_world_net
from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, make_mini_batch, convert_to_tensor

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


class Causal_world(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args, causal_graph):
        super(Causal_world,self).__init__()
        self.args = args        
        self.data = ReplayBuffer(action_prob_exist = True, max_size = int(self.args.traj_length), state_dim = state_dim, num_action = action_dim)
        self.w_model = Causal_world_net(self.args.layer_num, state_dim + action_dim, state_dim, \
                             self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
        self.w_model_optimizer = optim.Adam(self.w_model.parameters(), lr=self.args.critic_lr)

        self.error_cal = nn.MSELoss()
        self.model_loss = nn.MSELoss()

        self.writer = writer
        self.device = device
        self.causal_graph = causal_graph
          
    def v(self, x):
        pred_ = []
        for i in range(self.causal_graph.shape[0]):
            if len(x.shape) > 1:
                temp_graph = np.repeat(self.causal_graph[i].reshape(1, -1), repeats=x.shape[0], axis=0)
                graph_i = torch.from_numpy(temp_graph).bool().to(self.device)
            else:
                graph_i = torch.from_numpy(self.causal_graph[i]).bool().to(self.device)
            causal_x = torch.where(graph_i, x, 0)
            pred_.append(self.w_model(causal_x, iindex=i))
        return torch.hstack(tuple(pred_))
    
    def put_data(self,transition):
        self.data.put_data(transition)
    
    def get_error(self, state_action, next_state):
        pred_next_state = self.v(state_action).detach()
        error = self.error_cal(pred_next_state, next_state)
        return error.detach().item()

    def train_model(self,n_epi):
        data = self.data.sample(shuffle = False)
        state_actions, next_states = convert_to_tensor(self.device, np.hstack((data['state'], data['action'])), data['next_state'])

        for i in range(self.args.train_epoch):
            for state_action, next_state in make_mini_batch(self.args.batch_size, state_actions, next_states):
                pred_next_state = self.v(state_action)
                model_loss = self.model_loss(pred_next_state, next_state)

                self.w_model_optimizer.zero_grad()
                model_loss.backward()
                nn.utils.clip_grad_norm_(self.w_model.parameters(), self.args.max_grad_norm)
                self.w_model_optimizer.step()

                if self.writer != None:
                    self.writer.add_scalar("loss/world_model_loss", model_loss.item(), n_epi)
