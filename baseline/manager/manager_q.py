from mo.manager.manager import AbstractManager
from mo.examples.options_examples import OptionQArray
from mo.options.options import AbstractOption
from mo.examples.policy_examples_manager import QGraph
import numpy as np


class AgentQMontezuma(AbstractManager):

    def compute_total_reward(self, o_r_d_i, option_index, train_episode):
        """
        todo
        :param o_r_d_i:
        :param option_index:
        :param train_episode:
        :return:
        """
        reward = self.option_list[option_index].score
        if reward > 0:
            print("reward !")
        return reward

    def check_end_agent(self, o_r_d_i, current_option, train_episode):
        return o_r_d_i[-1]['ale.lives'] != 6

    def new_option(self) -> AbstractOption:
        return Option(self.action_space, self.parameters, self.get_number_options())

    def new_policy(self):
        return Policy(self.parameters)


class Option(OptionQArray):

    def compute_total_reward(self, o_r_d_i, intra_reward, action, end_option):
        """
        test ok
        :param o_r_d_i:
        :param intra_reward:
        :param action:
        :param end_option:
        :return:
        """
        total_reward = super().compute_total_reward(o_r_d_i, intra_reward, action, end_option)
        total_reward += (action != 0) * self.parameters["penalty_option_action"]
        total_reward += (action == 0) * self.parameters["penalty_option_idle"]
        total_reward += o_r_d_i[2] * self.parameters["penalty_death_option"]

        return total_reward


class Policy(QGraph):

    def find_best_action(self, train_episode=None):
        if not self.state_graph[self.current_state_index]:  # no alternative: explore
            return None, None

        if (train_episode is not None) and (np.random.rand() < self.parameters["probability_random_action_agent"]):
            return None, None

        else:
            # act differently if all actions have the same value
            if np.isclose(self.values[self.current_state_index],
                          self.values[self.current_state_index][0], atol=0.0).all():

                # todo change random behaviour !
                option_index = np.random.randint(len(self.values[self.current_state_index]))
                state_index = self.state_graph[self.current_state_index][option_index]
                return option_index, self.states[state_index]

            # else, return the regular best action
            else:
                return super().find_best_action(train_episode)

    def _update_states(self, state):
        novel = True
        if novel:
            super()._update_states(state)

        else:
            pass
