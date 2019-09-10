import numpy as np
from mo.options.options import AbstractOption
from a2c.utils.models import A2CEager
from a2c.utils.utils import ExperienceReplay


class A2COption(AbstractOption):

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
        self.main_model_nn = A2CEager(32,
                                      self.action_size,
                                      'A2Cnetwork',
                                      self.parameters["DEVICE"],
                                      self.parameters["CRITIC_NETWORK"],
                                      self.parameters["ACTOR_NETWORK"],
                                      self.parameters["LEARNING_RATE"],
                                      self.parameters["SHARED_CONVOLUTION_LAYERS"])

        self.gamma_min = self.parameters["GAMMA_MIN"]
        self.gamma_max = self.parameters["GAMMA_MAX"]
        self.batch_size = self.parameters["BATCH_SIZE"]
        self.weight_ce_exploration = self.parameters["WEIGHT_CE_EXPLORATION"]
        self.buffer = ExperienceReplay(self.batch_size)
        self.state = None
        self.number_of_lives = None

        print("NUMBER OF OPTIONS DISCOVERED: ", self.index)

    def _get_actor_critic_error(self, batch, train_episode):

        states_t = np.array([o[1][0] for o in batch])
        p = self.main_model_nn.prediction_critic(states_t)[:, 0]
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

        y_critic, adv_actor = self._returns_advantages(rewards, dones, p, p_, train_episode)

        print(y_critic)

        y_critic = np.expand_dims(y_critic, axis=-1)

        # print(self.option_id, y_critic, rewards, adv_actor, dones)

        return states_t, adv_actor, a_one_hot, y_critic

    def _returns_advantages(self, rewards, dones, values, next_value, train_episode):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        gamma = self.compute_current_gamma(train_episode)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def act(self, train_episode):
        predict = self.main_model_nn.prediction_actor(self.state)[0]
        return np.random.choice(self.action_space, p=predict)

    def replay(self, train_episode):

        if self.buffer.buffer_len() >= self.batch_size:
            batch = self.buffer.sample(self.batch_size, False)

            x, adv_actor, a_one_hot, y_critic = self._get_actor_critic_error(batch, train_episode)

            self.main_model_nn.train(x, y_critic, a_one_hot, adv_actor, self.weight_ce_exploration)

            self.buffer.reset_buffer()

    def reset(self, state):
        self.state = np.array([state])
        # self.buffer.reset_buffer()
        self.score = 0

    def get_critic_value(self, state):
        # get value of a state from the critic
        value = self.main_model_nn.prediction_critic([state])
        return value[0][0]

    def compute_current_gamma(self, train_episode):
        if self.parameters["EVOLUTION"] == "linear":
            return (self.gamma_max - self.gamma_min) / self.parameters["max_number_actions"] * train_episode + \
                   self.gamma_min

        if self.parameters["EVOLUTION"] == "static":
            return self.gamma_max

        else:
            raise NotImplementedError("Evolution of Gamma not implemented")

    def compute_total_reward(self, o_r_d_i, correct_termination, next_option_critic):
        total_reward = o_r_d_i[1]
        total_reward += self.compute_goal_reward(correct_termination)
        total_reward += self.lost_life(o_r_d_i[3])
        if next_option_critic is not None:
            total_reward += next_option_critic

        return total_reward

    def update_option(self, o_r_d_i, action, correct_termination, next_option_critic=None, train_episode=None):
        self.score += o_r_d_i[1]

        if train_episode:
            total_reward = self.compute_total_reward(o_r_d_i, correct_termination, next_option_critic)
            self.buffer.add((self.state[0], action, total_reward, o_r_d_i[0]["option"], o_r_d_i[2]))

        self.state = np.array([o_r_d_i[0]["option"]])
        self.replay(train_episode)

    def lost_life(self, info):
        """
        only for Montezuma : give the agent a penalty when it dies.
        :param info:
        :return:
        """
        if self.parameters["env_name"] == "MontezumaRevenge-v0":
            lost_life = self.number_of_lives != info["ale.lives"]
            self.number_of_lives = info["ale.lives"]

            return self.parameters["penalty_death_option"] if lost_life else 0

        else:
            return 0
