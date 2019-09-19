import numpy as np
from abc import ABCMeta, abstractmethod


class AbstractOptionExplore(metaclass=ABCMeta):

    def __init__(self, action_space):
        self.action_space = action_space
        self.index = None
        self.score = 0  # random option can find rewards too

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    def reset(self):
        """
        reset the reward
        :return: void
        """
        self.score = 0

    @abstractmethod
    def update_option(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def act(self, *args, **kwargs):
        raise NotImplementedError()


class OptionRandomExplore(AbstractOptionExplore):
    """
    This is an option that explores randomly
    """

    def __str__(self):
        return "explore option"

    def update_option(self, o_r_d_i, action, correct_termination, intra_reward=0, train_episode=None):
        """
        just update the reward variable (reward found by random explore)
        :return:
        """
        self.score += o_r_d_i[1]

    def act(self, train_episode):
        """
        :return: a random action from the action space
        """
        return np.random.choice(self.action_space)
