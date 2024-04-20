import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras import Input, layers, Model


class WorldModel(Model):
    def __init__(self, state_dim=2, action_dim=1, enviroment_model='linear'):
        super(WorldModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_size = 512
        self.batch_size = 64
        self.memory = np.zeros((self.memory_size, 2 * state_dim + action_dim))
        self.lr = 0.002
        self.set_causal_graph(enviroment_model)

    def set_causal_graph(self, env_model='linear'):
        if env_model == 'linear':
            self.shared_causal_net = self.add_linear_layer(self.state_dim + self.action_dim, self.state_dim)
        else:
            self.shared_causal_net = self.add_nonlinear_layer(self.state_dim + self.action_dim, self.state_dim)

    def add_linear_layer(self, input_dim, output_dim=1, lr=None, share_net=False):
        if lr is None:
            lr = self.lr
        if share_net:
            output_tensor = layers.Dense(output_dim, use_bias=False)(self.common_layer)
            model = Model(self.common_input_tensor, output_tensor)
        else:
            input_tensor = Input(shape=(input_dim,))
            x = layers.Dense(32, use_bias=False)(input_tensor)
            output_tensor = layers.Dense(output_dim, use_bias=False)(x)
            model = Model(input_tensor, output_tensor)
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def add_nonlinear_layer(self, input_dim, output_dim=1, lr=None, share_net=False):
        if lr is None:
            lr = self.lr
        if share_net:
            output_tensor = layers.Dense(output_dim, activation=tf.nn.sigmoid, use_bias=False)(self.common_layer)
            model = Model(self.common_input_tensor, output_tensor)
        else:
            input_tensor = Input(shape=(input_dim,))
            x1 = layers.Dense(32, activation=tf.nn.relu, use_bias=False)(input_tensor)
            x =  layers.Dense(8, activation=tf.nn.relu,use_bias=False)(x1)
            output_tensor = layers.Dense(output_dim, activation=tf.nn.sigmoid, use_bias=False)(x)
            model = Model(input_tensor, output_tensor)
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def set_common_net(self, input_dim, env_model='linear'):
        self.common_input_tensor = Input(shape=(input_dim,))
        if env_model == 'linear':
            self.common_layer = self.shared_causal_net.layers[-2](self.common_input_tensor)
        else:
            self.common_layer1 = self.shared_causal_net.layers[-3](self.common_input_tensor)
            self.common_layer = self.shared_causal_net.layers[-2](self.common_layer1)

    def set_net(self, causal_graph, env_model='linear'):
        self.set_common_net(self.state_dim + self.action_dim, env_model)
        causal_net = []
        for i in range(self.state_dim):
            if env_model == 'linear':
                causal_net.append(self.add_linear_layer(self.state_dim + self.action_dim, share_net=True))
            else:
                causal_net.append(self.add_nonlinear_layer(self.state_dim + self.action_dim, share_net=True))
            causal_net[i].layers[-1].set_weights(self.shared_causal_net.layers[-1].get_weights()[0][:,i].reshape(1,-1,1))                
        self.causal_net = causal_net
        self.causal_graph = causal_graph

    def store_data(self, o, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((o, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train(self, has_factorize=True):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        if has_factorize:
            for i in range(self.state_dim):
                id_nonzero = np.repeat(self.causal_graph[i].reshape(1, -1), repeats=self.batch_size, axis=0)
                
                fact_input_batch = np.where(id_nonzero, batch_memory[:, : self.state_dim + self.action_dim], 0)
                
                self.causal_net[i].fit(fact_input_batch, batch_memory[:, i - self.state_dim].reshape(self.batch_size, 1), verbose=0)
        else:
            self.shared_causal_net.fit(batch_memory[:, : self.state_dim + self.action_dim], batch_memory[:, self.state_dim + self.action_dim:], verbose=0)

    def predict_next_state(self, input_data, has_factorize=True):
        if not has_factorize:
            return self.shared_causal_net(input_data).numpy().reshape(-1)
        pred_fact_next_state = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            input_numpy = np.where(self.causal_graph[i], input_data, 0)
            fact_input = tf.constant(input_numpy)
            pred_fact_next_state[i] = self.causal_net[i](fact_input).numpy().reshape(-1)
        return pred_fact_next_state

    def get_reward(self, input_data, next_state, has_factorize=True):
        if not has_factorize:
            pred_next_state = self.shared_causal_net(input_data).numpy().reshape(-1)
            return tf.losses.mean_squared_error(pred_next_state, next_state).numpy().reshape(-1)
        causal_reward = np.zeros(1)
        for i in range(self.state_dim):
            input_numpy = np.where(self.causal_graph[i], input_data, 0)
            fact_input = tf.constant(input_numpy)
            fact_next_state = next_state[i].reshape(-1)
            pred_fact_next_state = self.causal_net[i](fact_input).numpy().reshape(-1)
            causal_reward += tf.losses.mean_squared_error(pred_fact_next_state, fact_next_state).numpy()
        return causal_reward / self.state_dim
    
    def call(self, input_data, has_factorize, idx=None):        
        if has_factorize:
            assert idx is not None
            id_nonzero = np.repeat(self.causal_graph[idx].reshape(1, -1), repeats=input_data.shape[0], axis=0)
            fact_input = np.where(id_nonzero, input_data, 0)
            return self.causal_net[idx](fact_input)
        else:
            return self.shared_causal_net(input_data)