from mo.options.options import AbstractOption
from mo.utils.miscellaneous import find_element_in_list
import numpy as np


class OptionQLearning(AbstractOption):
    """
    Example of a possible option
    _ This class is used when the number of states is unknown but the number of actions is known and discrete
    _ state_list = [state_1, state_2, ...] # a list of all states
    _ Action is an integer between 0 and number_actions - 1
    """

    def __init__(self, action_space, parameters, index):
        """
        :param action_space: a list of integers
        :param parameters: a dictionary containing the following keys
        "probability_random_action_option"
        "random_decay"
        "discount_factor"
        :param index: the index of the option
        :return:
        """
        super().__init__(action_space, parameters, index)
        self.number_actions = len(action_space)
        self.values = list()
        self.state_list = list()
        self.len_state_list = 0
        self.current_state_index = None

    def __str__(self):
        message = ""
        for idx in range(len(self.state_list)):
            message += str(self.state_list[idx]) + \
                       " values: " + str(self.values[idx]) + \
                       "\n"

        return message

    def act(self, train_episode=None):
        """
        An epsilon-greedy policy based on the values of self.policy (Q function)
        :param train_episode: if not None -> training phase. If None -> test phase
        :return: an action at the lower level
        """
        if (train_episode is not None) and self.exploration_needed(train_episode):
            return np.random.randint(self.number_actions)

        else:
            return np.argmax(self.values[self.current_state_index])

    def exploration_needed(self, train_episode):
        # either the random can be triggered by flipping a coin
        rdm = np.random.rand() < self.parameters["probability_random_action_option"] * \
            np.exp(-train_episode * self.parameters["random_decay"])

        # or if all values are equal
        equal_values = np.all(self.values[self.current_state_index][0] == self.values[self.current_state_index])

        return rdm or equal_values

    def reset(self, state):
        """
        Reset the current state
        :return: void
        """
        self.score = 0
        self._update_states(state)

    def update_option(self, o_r_d_i, action, correct_termination, train_episode=None):
        # compute the rewards
        total_reward = o_r_d_i[1] + self.compute_goal_reward(correct_termination)

        # update the parameters
        end_option = correct_termination is not None
        new_state = o_r_d_i[0]["option"]
        self._update_values(new_state, total_reward, action, end_option, train_episode)
        self._update_states(new_state)

        # compute the total score
        self.score += o_r_d_i[1]

    def _update_states(self, next_state):
        next_state_index = find_element_in_list(next_state, self.state_list)

        if next_state_index is not None:  # next_state already in state_list
            self.current_state_index = next_state_index

        else:  # next_state is a new state
            self.state_list.append(next_state)
            self.values.append(np.zeros(self.number_actions, dtype=np.float64))
            self.len_state_list += 1
            next_state_index = self.len_state_list - 1
            self.current_state_index = next_state_index

    def _update_values(self, new_state, reward, action, end_option, train_episode):
        """
        updates the values of the policy.
        Only updates the current state in the simulation phase which corresponds to train_episode == None.
        :param new_state:
        :param reward:
        :param action:
        :param end_option:
        :param train_episode:
        type == None -> simulation,
        type == integer -> training.
        :return: void
        """
        if train_episode is not None:  # training phase: update value
            if end_option:
                best_value = 0
            else:
                new_state_index = find_element_in_list(new_state, self.state_list)
                if new_state_index is not None:
                    best_value = np.max(self.values[new_state_index])

                else:
                    best_value = 0

            self.values[self.current_state_index][action] *= (1 - self.parameters["discount_factor"])
            self.values[self.current_state_index][action] += self.parameters["discount_factor"] * (reward + best_value)
