from mo.options.options import AbstractOption
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from mo.manager.manager import AbstractManager
from mo.options.options_explore import AbstractOptionExplore
from mo.policies.policy_manager import AbstractPolicyManager
from mo.options.options_explore import OptionRandomExplore


class AgentDQN(AbstractManager):

    def new_explore_option(self) -> AbstractOptionExplore:
        return OptionRandomExplore(self.action_space)

    def new_policy(self) -> AbstractPolicyManager:
        return AbstractPolicyManager()

    def new_option(self) -> AbstractOption:
        return DQNOption(self.action_space, self.parameters, self.get_number_options())


class DQNOption(AbstractOption):

    def __init__(self, action_space, parameters, index):
        super().__init__(action_space, parameters, index)
        self.state_size = 1  # gives the hashed image
        self.action_size = len(action_space)
        self.memory = deque(maxlen=2000)
        self.epsilon = self.parameters["epsilon"]  # exploration rate
        self.epsilon_min = self.parameters["epsilon_min"]
        self.epsilon_decay = self.parameters["epsilon_decay"]
        self.learning_rate = self.parameters["learning_rate"]
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

    def act(self, train_episode):
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
                q_next = self.model.predict(next_state)[0]
                target = (reward + self.learning_rate * np.amax(q_next))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            # train network
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def reset(self, initial_state, current_state, terminal_state):
        self.state = np.array([current_state])

    def compute_total_score(self, o_r_d_i, action, correct_termination):
        return self.score + o_r_d_i[1]

    def update_option_policy(self, state, total_reward, action, correct_termination, train_episode):
        self.memory.append((self.state, action, total_reward, state, correct_termination))
        self.state = np.array([state])

    def compute_total_reward(self, o_r_d_i, correct_termination):
        total_reward = o_r_d_i[1]
        total_reward += self.compute_goal_reward(correct_termination)
        total_reward += o_r_d_i[2] * self.parameters["penalty_death_option"]
        return total_reward

    def get_value(self, state):
        act_values = self.model.predict(state)
        return max(act_values)
