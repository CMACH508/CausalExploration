# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np
from keras import Input, layers, Model, models
from keras.optimizers import Adam


class World_model(Model):
    def __init__(self, state_dim=38, action_dim=1):
        super(World_model, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.update_outdated = 0
        self.memory_size = 1024
        self.FIRST_UPDATE = 100        
        self.UPDATE_PERIOD = 20                                               
        self.batch_size = 64
        self.lr = 0.002        
        self.memory = np.zeros((
            self.memory_size,
            2 * state_dim + action_dim
        ))

    def _build_shared(self, input_dim):
        self._input = Input(shape=(input_dim,))
        self._hd1 = layers.Dense(32, activation=tf.nn.relu, use_bias=False)(self._input)
        self._hd2 = layers.Dense(8, activation=tf.nn.relu,use_bias=False)(self._hd1)
    
    def _build_specific(self, input_dim, output_dim=1, lr=None):
        if lr is None:
            lr = self.lr
        _output = layers.Dense(output_dim, activation=tf.nn.sigmoid, use_bias=False)(self._hd2)
        model = Model(self._input, _output)
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        return model

    def _build(self, causal_graph):
        self._build_shared(self.state_dim + self.action_dim)
        causal_net = []
        for i in range(self.state_dim):            
            causal_net.append(self._build_specific(self.state_dim + self.action_dim))                
        self.causal_net = causal_net
        self.causal_graph = causal_graph

    def store_data(self, o, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((o, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train(self, current_time):
        if (current_time < self.FIRST_UPDATE) or (current_time - self.update_outdated < self.UPDATE_PERIOD):
            return

        self.update_outdated = current_time

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        for i in range(self.state_dim):
            id_nonzero = np.repeat(self.causal_graph[i].reshape(1, -1), repeats=self.batch_size, axis=0)            
            fact_input_batch = np.where(id_nonzero, batch_memory[:, : self.state_dim + self.action_dim], 0)                
            self.causal_net[i].fit(fact_input_batch, batch_memory[:, i - self.state_dim].reshape(self.batch_size, 1), verbose=0)

    def get_reward(self, input_data, next_state):
        causal_reward = 0
        for i in range(self.state_dim):
            input_numpy = np.where(self.causal_graph[i], input_data, 0)
            fact_next_state = next_state.reshape(-1)[i]
            pred_fact_next_state = self.causal_net[i].predict(input_numpy, verbose=0).reshape(-1)
            with tf.Session() as sess:
                causal_reward += tf.losses.mean_squared_error(pred_fact_next_state, fact_next_state.reshape(-1)).eval()
        return causal_reward / self.state_dim

    def save_net(self):
        for i in range(self.state_dim):
            self.causal_net[i].save('./world_model/state_{}.h5'.format(str(i)))

    def load_net(self, causal_graph):
        causal_net = []
        for i in range(self.state_dim):
            model_path = "./world_model/state_{}.h5".format(str(i))
            causal_net.append(models.load_model(model_path))
        
        self.causal_net = causal_net
        self.causal_graph = causal_graph
