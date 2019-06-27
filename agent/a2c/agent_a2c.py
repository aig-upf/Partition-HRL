from ao.options.options import OptionAbstract
import numpy as np
from agent.agent import AgentQMontezuma
from agent.a2c.utils import ExperienceReplay
from agent.a2c.models import A2CEager


class AgentA2C(AgentQMontezuma):

    def __init__(self, action_space, parameters):
        super().__init__(action_space, parameters)

        # this parameter is useful if you want to restart the episode after n actions (200 actions for instance)
        self.nb_actions = 0

    def reset(self, initial_state):
        print(len(self))
        self.nb_actions = 0
        super().reset(initial_state)

    def check_end_agent(self, o_r_d_i, current_option, train_episode):
        self.nb_actions += 1
        return super().check_end_agent(o_r_d_i, current_option, train_episode) or \
               bool(self.nb_actions > self.parameters["max_number_actions"])

    def get_option(self) -> OptionAbstract:
        return OptionA2C(self.action_space, self.parameters, len(self), )

    def update_agent(self, o_r_d_i, option, train_episode=None):
        super().update_agent(o_r_d_i, option, train_episode)


class OptionA2C(OptionAbstract):

    def __init__(self, action_space, parameters, index):
        super().__init__(action_space, parameters, index)

        # not the right shape here
        self.input_shape_nn = [None, self.parameters["NUMBER_ZONES_OPTION_Y"],
                               self.parameters["NUMBER_ZONES_OPTION_X"],
                               self.parameters["stack_images_length"]]

        self.state_size = self.input_shape_nn[1:]
        self.state_dimension = tuple(self.state_size)

        self.action_size = len(action_space)
        # shared variable
        self.main_model_nn = A2CEager(self.input_shape_nn,
                                      32,
                                      self.action_size,
                                      'A2Cnetwork',
                                      self.parameters["DEVICE"],
                                      self.parameters["CRITIC_NETWORK"],
                                      self.parameters["ACTOR_NETWORK"],
                                      self.parameters["LEARNING_RATE_ACTOR"],
                                      self.parameters["LEARNING_RATE_CRITIC"],
                                      self.parameters["SHARED_CONVOLUTION_LAYERS"])

        self.gamma = self.parameters["GAMMA"]
        self.learning_rate_actor = self.parameters["LEARNING_RATE_ACTOR"]
        self.learning_rate_critic = self.parameters["LEARNING_RATE_CRITIC"]
        self.batch_size = self.parameters["BATCH_SIZE"]
        self.weight_ce_exploration = self.parameters["WEIGHT_CE_EXPLORATION"]
        self.buffer = ExperienceReplay(self.batch_size)
        self.state = None

    def _get_actor_critic_error(self, batch):

        states_t = np.array([o[1][0] for o in batch])
        p = self.main_model_nn.prediction_critic(states_t)
        a_one_hot = np.zeros((len(batch), len(self.action_space)))
        dones = np.zeros((len(batch)))
        rewards = np.zeros((len(batch)))

        for i in range(len(batch)):
            o = batch[i][1]
            a = o[1]
            r = o[2]
            s_ = o[3]

            a_index = self.action_space.index(a)

            if s_ is None:
                dones[i] = 1
                p_ = [0]
            elif i == len(batch)-1:
                p_ = self.main_model_nn.prediction_critic([s_])[0]

            rewards[i] = r
            a_one_hot[i][a_index] = 1

        y_critic, adv_actor = self._returns_advantages(rewards, dones, p, p_)
        y_critic = np.expand_dims(y_critic, axis=-1)

        return states_t, adv_actor, a_one_hot, y_critic

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def act(self, train_episode):

        predict = self.main_model_nn.prediction_actor(self.state)[0]

        return np.random.choice(self.action_space, p=predict)

    def replay(self):

        if self.buffer.buffer_len() >= self.batch_size:
            batch = self.buffer.sample(self.batch_size, False)

            x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch)

            self.main_model_nn.train_actor(x, a_one_hot, adv_actor, self.weight_ce_exploration)
            self.main_model_nn.train_critic(x, y_critic)

            self.buffer.reset_buffer()

    def update_option(self, o_r_d_i, intra_reward, action, end_option, train_episode):
        r = self.compute_total_reward(o_r_d_i, action, intra_reward, end_option)

        if train_episode:

            self.buffer.add((self.state[0], action, r, o_r_d_i[0]["option"], o_r_d_i[2]))

        self.state = np.array([o_r_d_i[0]["option"]])
        self.replay()

    def reset(self, initial_state, current_state, terminal_state):

        super().reset_states(initial_state, terminal_state)
        self.state = np.array([current_state])
        self.buffer.reset_buffer()

    def compute_total_score(self, *args, **kwargs):
        pass

    def compute_total_reward(self, o_r_d_i, action, intra_reward, end_option):
        """
        test ok
        :param o_r_d_i:
        :param action:
        :param intra_reward:
        :param end_option:
        :return:
        """
        total_reward = o_r_d_i[1] + intra_reward
        if end_option:
            total_reward += (self.terminal_state == o_r_d_i[0]["agent"]) * self.parameters["reward_end_option"]
            total_reward += (self.terminal_state != o_r_d_i[0]["agent"]) * self.parameters["penalty_end_option"]

        total_reward += self.parameters["penalty_option_action"]
        total_reward += o_r_d_i[2] * self.parameters["penalty_death_option"]

        return total_reward

    def get_value(self, state):

        value = self.main_model_nn.prediction_critic([state])

        return value
