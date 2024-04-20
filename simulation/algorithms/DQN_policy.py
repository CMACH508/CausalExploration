import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import layers, Model, Input


class DQN:
    def __init__(self, state_dim, action_dim, action_range):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.lr = 0.001
        self.gamma = 0.9
        self.replace_target_iter = 10
        self.memory_size = 512
        self.batch_size = 64
        self.epsilon_max = 0.99
        self.epsilon_increment = 0.05
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, state_dim*2 + self.action_dim + 1))
        self.model_eval = self.create_eval(action_dim * action_range, state_dim)
        self.model_target = self.create_target(action_dim * action_range, state_dim)
        self.model_eval.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
        self.cost_his = []

    def create_eval(self, action_range, state_dim):
        input_tensor = Input(shape=(state_dim,))
        x = layers.Dense(32, activation='relu')(input_tensor)
        x = layers.Dense(32, activation='relu')(x)
        output_tensor = layers.Dense(action_range)(x)
        model = Model(input_tensor, output_tensor)
        # model.summary()
        return model

    def create_target(self, action_range, state_dim):
        input_tensor = Input(shape=(state_dim,))
        x = layers.Dense(32, activation='relu', trainable=False)(input_tensor)
        x = layers.Dense(32, activation='relu', trainable=False)(x)
        output_tensor = layers.Dense(action_range)(x)
        model = Model(input_tensor, output_tensor)
        # model.summary()
        return model

    def store_transition(self, s, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, r, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            action_value = self.model_eval(observation)
            if self.action_dim == 1:
                action = np.argmax(action_value)
            else:
                new_value = action_value.numpy().reshape(self.action_dim, -1)
                action = np.argmax(new_value, axis=1)
        else:
            action = np.random.randint(0, self.action_range, size=self.action_dim)
        return action / (self.action_range - 1)

    def _replace_target_params(self):
        for eval_layer, target_layer in zip(self.model_eval.layers, self.model_target.layers):
            target_layer.set_weights(eval_layer.get_weights())

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]  # shuffle
        q_next = self.model_target.predict(batch_memory[:, -self.state_dim:], verbose=0)
        q_eval = self.model_eval.predict(batch_memory[:, :self.state_dim], verbose=0)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.state_dim].astype(int)

        reward = batch_memory[:, self.state_dim + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()

        cost = self.model_eval.train_on_batch(batch_memory[:, :self.state_dim], q_target)
        self.cost_his.append(cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('mse')
        plt.xlabel('training steps')
        plt.show()
