import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model, Input, initializers


def init_env_mask(input_dim=2, node_dim=10):
    """
        generate a constant kernel for dense network
            :param input_dim: input shape of layer
            :param node_dim: units of layer
            :return: kernel
    """
    g_random = np.random.uniform(low=-0.8, high=0.8, size=(input_dim, node_dim))
    g_random[np.random.randn(input_dim, node_dim) < 0] = 0

    g_random = np.tril(g_random, -1)

    W = g_random

    return W


class GroundEnv(Model):
    def __init__(self, state_dim=2, action_dim=1, model_type='linear'):
        super(GroundEnv, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.z_1 = np.random.multivariate_normal(np.zeros(state_dim), 0.01 * np.eye(state_dim))

        self.cov_mat_t = np.diag(np.random.uniform(0, 0.01, state_dim))
        
        self.model_type = model_type

        if model_type == 'linear':
            self.mean_net = self.build_linear_model(state_dim + action_dim, state_dim)
        else:
            self.mean_net = self.build_nonlinear_model(state_dim + action_dim, state_dim)

        self.true_weight = self.mean_net.get_weights()[0]
        self.causal_mask = np.where(self.true_weight != 0, 1, 0).T

    def change_causal(self):
        for _ in range(self.state_dim):
            random_change = np.random.randint(0,2)
            random_x_index = np.random.randint(0,self.state_dim + self.action_dim)
            random_y_index = np.random.randint(0,self.state_dim)
            if random_change:
                self.true_weight[random_x_index, random_y_index] = np.random.uniform(low=-0.8, high=0.8)
            else:
                self.true_weight[random_x_index, random_y_index] = 0

        self.causal_mask = np.where(self.true_weight != 0, 1, 0).T
        self.mean_net.set_weights(np.expand_dims(self.true_weight, axis=0))
            
    def build_linear_model(self, input_dim, output_dim):
        input_tensor = Input(shape=(input_dim,))
        output_tensor = layers.Dense(output_dim, use_bias=False, kernel_initializer=tf.constant_initializer(init_env_mask(input_dim, output_dim)), trainable=False)(input_tensor)
        model = Model(input_tensor, output_tensor)
        return model

    def build_nonlinear_model(self, input_dim, output_dim):
        input_tensor = Input(shape=(input_dim,))
        output_tensor = layers.Dense(output_dim, activation=tf.nn.sigmoid, use_bias=False, kernel_initializer=tf.constant_initializer(init_env_mask(input_dim, output_dim)), trainable=False)(input_tensor)
        model = Model(input_tensor, output_tensor)
        return model

    def call(self, input_t1):
        if self.model_type == 'linear':
            z_t = self.mean_net(input_t1).numpy().reshape(-1)
        else:
            mean_t = self.mean_net(input_t1).numpy().reshape(-1)
            z_t = np.random.multivariate_normal(mean_t, self.cov_mat_t)        
        return z_t