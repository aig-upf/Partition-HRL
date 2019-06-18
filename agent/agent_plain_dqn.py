from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from ao.examples.agent_example import PlainQLearning
import numpy as np


class PlainDQNMontezuma(PlainQLearning):

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)
        self.policy = None

        self.state_size = 1  # gives the hashed image
        self.action_size = len(action_space)
        self.memory = deque(maxlen=parameters["memory"])
        self.gamma = parameters["gamma"]  # discount rate
        self.epsilon = parameters["epsilon"]  # exploration rate
        self.epsilon_min = parameters["epsilon_min"]
        self.epsilon_decay = parameters["epsilon_decay"]
        self.learning_rate = parameters["learning_rate"]
        self.model = self._build_model()
        self.state = None

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, train_episode=None):
        if train_episode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # select random action with prob=epsilon else action=maxQ

            if np.random.rand() <= self.epsilon:
                return np.random.randint(self.action_size)

        act_values = self.model.predict(self.state)  # [[action 1, action 2]]
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # sample random transitions
        choices = np.array(self.memory)
        idx = np.random.choice(len(choices), batch_size)
        minibatch = choices[idx]

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                """
                Q* satisfies the bellman equation: Q(s,a) = r + gamma * Q(s',pi(s'))
                """
                Q_next = self.model.predict(next_state)[0]
                target = (reward + self.gamma * np.amax(Q_next))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            # train network
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def check_end_agent(self, o_r_d_i):
        return o_r_d_i[3]["ale.lives"] != 6

    def update_agent(self, o_r_d_i, action, train_episode=None):
        if train_episode:
            self.memory.append((self.state, action, o_r_d_i[1], self.my_hash(o_r_d_i[0]), o_r_d_i[2]))
            self.state = np.array([self.my_hash(o_r_d_i[0])])

        self.score += self.compute_total_score(o_r_d_i)

    def reset(self, current_state):
        self.state = np.array([self.my_hash(current_state)])

    def compute_total_reward(self, o_r_d_i, train_episode):
        total_reward = o_r_d_i[1]
        total_reward += o_r_d_i[2] * self.parameters["penalty_lost_life_for_agent"]

        return total_reward

    def my_hash(self, observation):
        observation_tuple = tuple(tuple(tuple(color) for color in lig) for lig in observation)
        return hash(observation_tuple)