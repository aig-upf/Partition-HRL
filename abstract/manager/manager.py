"""
This library can be connected to a gym environment or any kind of environment as long as it has the following methods:
- env.reset
- env.step
"""
import numpy as np

from abc import ABCMeta, abstractmethod
from tqdm import tqdm
from abstract.policies.policy_manager import AbstractPolicyManager
from abstract.options.options import AbstractOption
from abstract.options.options_explore import AbstractOptionExplore
from abstract.utils.save_results import SaveResults
from abstract.utils.show_render import ShowRender
from abstract.utils.miscellaneous import obs_equal, SSIM_obs_equal, constrained_type, check_type
from collections import deque


class AbstractManager(metaclass=ABCMeta):
    """
    Abstract Manager class.
    A Manager manages options (see Sutton's framework)
    """

    def __init__(self, action_space, parameters):
        """
        initialize the manager's parameters.
        :param action_space: the actions to be selected by the options.
        :param parameters: a dictionary containing the parameters of the experiment (for the agent and the environment).
        """
        self.action_space = action_space
        self.parameters = parameters
        self.option_list = []

        self.policy = self.new_policy()
        self.explore_option = self.new_explore_option()
        self.show_render = None

        self.save_results = SaveResults()
        self.deque_max_length = 100
        self.number_transitions_made = 0
        self.successful_transition = deque(maxlen=self.deque_max_length)  # A list of 0 and 1 of size <=100.
        self.score = deque(maxlen=self.deque_max_length)

        # checks that policy and options have the right type.
        #constrained_type(self.policy, AbstractPolicyManager)
        #constrained_type(self.explore_option, AbstractOptionExplore)

    def reset_all(self):
        # deleting all the keras subclassing models
        for options in self.option_list:
            del options

        self.parameters["SHARED_CONVOLUTION_LAYERS"].reset()

        self.option_list = []
        self.policy = self.new_policy()
        self.explore_option = self.new_explore_option()
        self.successful_transition = deque(maxlen=self.deque_max_length)
        self.score = deque(maxlen=self.deque_max_length)

    def reset(self, initial_state):
        self.score.append(0)
        self.policy.reset(initial_state)

    def _train_simulate(self, env, train_episode=None):
        """
        Method used to train or simulate the manager (the main loop)

        a) choose an option
        b) option acts and updates
        c) if a new state is found then update manager

        :param env: the environment.
        :param train_episode: the episode of training.
        - if not None: training
        - if None: simulating
        :return: void
        """
        # The initial observation
        o_r_d_i = [env.reset()] + [None]*3  # o_r_d_i means "Observation_Reward_Done_Info"
        # Reset all the manager parameters
        self.reset(o_r_d_i[0]["manager"])
        done = False
        current_option = None
        # Render the current state
        if self.parameters["display_environment"]:
            self.show_render.render(o_r_d_i[0])

        while not done:
            # If no option is activated then choose one
            if current_option is None:
                current_option = self.select_option(o_r_d_i, train_episode)
                assert current_option.score == 0, "the option's reset function must reset the score to 0."

            # choose an action
            action = current_option.act(train_episode)

            # make an action and display the state space
            o_r_d_i = env.step(action)
            if self.parameters["display_environment"]:
                self.show_render.render(o_r_d_i[0])

            # check if the option ended correctly
            correct_termination = self.check_end_option(current_option, o_r_d_i[0]["manager"])

            # update the option
            current_option.update_option(o_r_d_i, action, correct_termination, train_episode)

            # If the option is done, update the manager
            if correct_termination is not None:
                if check_type(current_option, AbstractOption):
                    # record the correct transition when the option is a regular option (i.e. not an explore option)
                    self.successful_transition.append(correct_termination)
                    self.write_success_rate_transitions()

                # the manager does not need to know if the correct_termination is 0 or 1.
                self.update_manager(o_r_d_i, current_option, train_episode)
                current_option = None

            done = self.check_end_manager(o_r_d_i)

        self.write_manager_score(train_episode)

    def select_option(self, o_r_d_i, train_episode=None):
        """
        The manager will
        1) choose an option in option_list
        2) reset the parameters of this option
        3) return the option
        :return: an option
        """
        best_option_index = self.policy.find_best_action(train_episode)
        if best_option_index is None:
            # in this case : explore
            self.explore_option.reset()
            return self.explore_option

        else:
            # set the option at the right position and return it
            self.option_list[best_option_index].reset(o_r_d_i[0]["option"])
            return self.option_list[best_option_index]

    def update_manager(self, o_r_d_i, option, train_episode=None):
        """
        updates the manager parameters.
        In simulation mode, updates
        - score
        In training mode, updates
        - policy
        - option
        :param o_r_d_i : Observation, Reward, Done, Info given by function step
        :param option: the index of the option that did the last action
        :param train_episode: the number of the current training episode
        :return : void
        """
        if train_episode is None:  # in simulation mode
            self.score[-1] += option.score

        else:  # in training mode
            self.score[-1] += option.score
            self._update_policy(o_r_d_i, option)

            # add a new option if necessary
            missing_option = self.compute_number_options_needed() - self.get_number_options()
            assert missing_option == 1 or missing_option == 0, "number of options is wrong"
            if missing_option:
                new_option = self.new_option()
                constrained_type(new_option, AbstractOption)
                self.option_list.append(new_option)

    def _update_policy(self, o_r_d_i, option):
        # print("option " + str(option.index) + " score = " + str(option.score))
        self.policy.update_policy(o_r_d_i[0]["manager"], option.score)

    def train(self, environment, parameters, seed=0):
        """
        Method used to train the RL manager. It calls function _train_simulate_manager with the current training episode
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        # prepare the file for the results
        self.save_results.set_seed(seed)
        self.save_results.write_setting(parameters)

        # prepare to display the states
        if self.parameters["display_environment"]:
            self.show_render = self.get_show_render_train()

        for t in tqdm(range(1, self.parameters["number_episodes"] + 1)):
            self._train_simulate(environment, t)

            if not t % self.parameters["episodes_performances"]:
                assert len(self.option_list) > 0, \
                    "no option found, probably because the agent does not see any new state. " \
                    "You should tune the parameter THRESH_BINARY_MANAGER or " \
                    "increase the number of zones for the manager."

                self.save_results.plot_success_rate_transitions()
                self.save_results.plot_manager_score()

        if self.parameters["display_environment"]:
            self.show_render.close()

    def simulate(self, environment, seed=0):
        """
        Method used to train the RL manager.
        It calls _train_simulate_manager method with parameter "train_episode" set to None
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        # prepare to display the states if needed
        if self.parameters["display_environment"]:
            self.show_render = self.get_show_render_simulate()

        # simulate
        self._train_simulate(environment)

        if self.parameters["display_environment"]:
            self.show_render.close()

    def write_success_rate_transitions(self):
        """
        Write in a file the sum of the last 100 transitions.
        A transition is 0 or 1.
        1 if the option terminates at the right abstract state and 0 otherwise.
        :return: void
        """
        self.number_transitions_made += 1
        if len(self.successful_transition) >= self.deque_max_length:
            self.save_results.write_message_in_a_file(self.save_results.success_rate_file_name,
                                                      str(self.number_transitions_made) + " " +
                                                      str(np.mean(self.successful_transition)*100) + "\n")

    def write_manager_score(self, train_episode):
        """
        Write in a file the manager's score.
        """
        if len(self.score) >= self.deque_max_length:
            self.save_results.write_message_in_a_file(self.save_results.manager_score_file_name,
                                                      str(train_episode) + " " + str(np.mean(self.score)) + "\n")

    def check_end_option(self, option, obs_manager):
        """
        check if the option ended and if the termination is correct.
        :param option: explore option or regular option
        :param obs_manager:
        :return:
        - None if the option is not done.
        Otherwise:
        if option is an explore option:
        - True
        if option is a regular option:
        - True if ended in the correct new abstract state, False if the new abstract state is wrong.
        """
        #if obs_equal(self.get_current_state(), obs_manager):
        if SSIM_obs_equal(self.get_current_state(), obs_manager, not self.parameters["GRAY_SCALE"]):
            # option is not done
            return None

        else:
            # option is done
            if check_type(option, AbstractOptionExplore):
                # print("explore")
                return True

            elif check_type(option, AbstractOption):
                correct_transition = obs_equal(obs_manager, self.get_terminal_state(option.index))
                # if correct_transition:
                #    print("correct final state")
                # else:
                #    print("wrong final state")
                return correct_transition

            else:
                raise Exception(type(option).__name__ + " is not supported")

    @staticmethod
    def get_show_render_train():
        return ShowRender()

    @staticmethod
    def get_show_render_simulate():
        return ShowRender()

    @staticmethod
    def check_end_manager(o_r_d_i):
        """
        Check if the current episode is over or not.
        The output of this function will update the variable "done" in method self._train_simulate_manager.
        Returns by default o_r_d_i[2], but can be overwritten.
        :param o_r_d_i:
        :return: True iff the manager is done.
        """
        return o_r_d_i[2]

    def get_number_options(self):
        return len(self.option_list)

    def compute_number_options_needed(self):
        return self.policy.get_max_number_successors()

    def get_current_state(self):
        return self.policy.get_current_state()

    def get_terminal_state(self, option_index):
        return self.policy.get_next_state(option_index)

    def get_result_paths(self):
        return {"manager": self.save_results.dir_path_seed + "/" + self.save_results.manager_score_file_name,
                "transitions": self.save_results.dir_path_seed + "/" + self.save_results.success_rate_file_name}

    def get_result_folder(self):
        return self.save_results.dir_path

    # Method to be implemented by the sub classes

    @abstractmethod
    def new_option(self) -> AbstractOption:
        """
        Make a new option to update the list: self.option_list
        This method depends on the kind of option we want to use.
        :return: a class which inherits from AbstractOption
        """
        raise NotImplementedError()

    @abstractmethod
    def new_explore_option(self) -> AbstractOptionExplore:
        """
        Make a new option explore
        :return: a class which inherits from AbstractOptionExplore
        """
        raise NotImplementedError()

    @abstractmethod
    def new_policy(self) -> AbstractPolicyManager:
        """
        make a new policy for the manager
        :return: a class which inherits from AbstractPolicyManager
        """
        raise NotImplementedError()
