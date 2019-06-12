from ao.examples.agent_example import AgentOptionMontezuma
from ao.examples.options_examples import OptionQArray
from ao.options.options import OptionAbstract
from ao.examples.policy_examples_agent import QGraph
import numpy as np


class Agent(AgentOptionMontezuma):

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

    def reset(self, initial_state):
        if self.get_current_state() != initial_state:
            super().reset(initial_state)

    def check_end_agent(self, o_r_d_i, current_option, train_episode):
        return o_r_d_i[-1]['ale.lives'] != 6

    def get_option(self) -> OptionAbstract:
        return Option(self.action_space, self.parameters, len(self))

    def get_policy(self):
        return Policy(self.parameters)

    def _train_simulate_agent(self, environment, train_episode=None):
        """
        Method used to train or simulate the agent

        a) choose an option
        b) option acts and updates
        c) if a new state is found then update agent

        :param environment:
        :param train_episode: the episode of training.
        :return:
        """
        # The initial observation
        print(len(self))
        obs = environment.reset()
        o_r_d_i = [obs]

        # Reset all the parameters
        self.reset(o_r_d_i[0]["agent"])
        done = False
        current_option = None

        # Render the current state
        self.display_state(environment, train_episode)

        while not done:
            # If no option is activated then choose one
            if current_option is None:
                current_option = self.act(o_r_d_i, train_episode)

            # choose an action
            action = current_option.act()

            # make an action and display the state space
            # todo record the learning curve
            o_r_d_i = environment.step(action)

            self.display_state(environment, train_episode)

            # update the option
            current_option.update_option(o_r_d_i, action, train_episode)

            # check if the option ended
            end_option = current_option.check_end_option(o_r_d_i[0]["agent"])

            done = self.check_end_agent(o_r_d_i, current_option, train_episode)

            # If the option is done, update the agent
            if end_option:
                self.update_agent(o_r_d_i, current_option, train_episode)
                current_option = None


class Option(OptionQArray):

    def update_option(self, o_r_d_i, action, train_episode=None):
        """
        updates the parameters of the option, in particular self.policy.
        Train mode and simulate mode are distinguished by the value of train_episode
        :param o_r_d_i:  Observation, Reward, Done, Info
        :param action: the last action performed
        :param train_episode: the number of the current training episode
        :return: void
        """
        # check if the option is done
        end_option = self.check_end_option(o_r_d_i[0]["agent"])

        if train_episode is not None:
            # compute the rewards
            total_reward = self.compute_total_reward(o_r_d_i, action, end_option)

            # update the q function
            self.policy.update_policy(o_r_d_i[0]["option"], total_reward, action, end_option)

        # compute the total score
        self.score = self.compute_total_score(o_r_d_i, action, end_option, train_episode)

    def compute_total_reward(self, o_r_d_i, action, end_option):
        """
        test ok
        :param o_r_d_i:
        :param action:
        :param end_option:
        :return:
        """
        total_reward = super().compute_total_reward(o_r_d_i, action, end_option)
        total_reward += (action != 0) * self.parameters["penalty_option_action"]
        total_reward += (action == 0) * self.parameters["penalty_option_action"] / 10
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

                option_index = np.argmax(self.values[self.current_state_index])
                state_index = self.state_graph[self.current_state_index][option_index]
                return option_index, self.states[state_index]

            # else, return the regular best action
            else:
                return super().find_best_action(train_episode)


class OptionDQN(OptionAbstract):
    """
    TODO
    """
    def update_option(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def compute_total_reward(self, *args, **kwargs):
        pass

    def compute_total_score(self, *args, **kwargs):
        pass