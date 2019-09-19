from abc import ABCMeta, abstractmethod


class AbstractOption(metaclass=ABCMeta):
    """
    Abstract option class.
    """

    def __init__(self, action_space, parameters, index):
        """
        initial_state: where the option starts;
        terminal_state: where the option has to go;
        type_policy: the name of the class used for the policy
        """
        self.action_space = action_space
        self.parameters = parameters
        self.score = 0
        self.index = index

    def __str__(self):
        return "option " + str(self.index)

    def compute_goal_reward(self, correct_termination):
        """
        A function that computes the reward or the penalty gotten when terminating.
        :param correct_termination:
        :return:
        """
        goal_reward = 0
        if correct_termination is None:
            return goal_reward

        if correct_termination:
            goal_reward += self.parameters["reward_end_option"]

        else:
            goal_reward += self.parameters["penalty_end_option"]

        return goal_reward

    # methods that have to be implemented by the sub classes.
    @abstractmethod
    def act(self, train_episode):
        """
        Performs an action
        :return: an integer in range(self.number_actions)
        """
        raise NotImplementedError()

    @abstractmethod
    def update_option(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, state):
        """
        reset parameters
        :return:
        """
        raise NotImplementedError()
