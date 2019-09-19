from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
from abstract.utils.show_render import ShowRender
from abstract.utils.save_results import SaveResults


class AbstractAgent(metaclass=ABCMeta):
    """
    Abstract Agent, this is *not* a manager, *no* options here.
    """

    def __init__(self, action_space, parameters):
        self.parameters = parameters
        self.action_space = action_space
        self.current_state = None
        self.score = 0
        self.show_render = None

    @abstractmethod
    def reset(self, initial_state):
        raise NotImplementedError()

    @abstractmethod
    def act(self, train_episode):
        raise NotImplementedError()

    @abstractmethod
    def update_agent(self, o_r_d_i, action, train_episode):
        raise NotImplementedError()

    def _train_simulate(self, env, train_episode=None):
        # reset the parameters
        obs = env.reset()
        self.reset(obs)
        done = False

        # render the image
        if self.parameters["display_environment"]:
            self.show_render.render(obs)

        while not done:
            # choose an action
            action = self.act(train_episode)

            # get the output
            o_r_d_i = env.step(action)

            # update the agent
            self.update_agent(o_r_d_i, action, train_episode)

            # display the observation if needed
            if self.parameters["display_environment"]:
                self.show_render.render(o_r_d_i[0])

            # update variable done
            done = self.check_end_agent(o_r_d_i)

    def train(self, environment, seed=0):
        """
        Method used to train the RL agent. It calls function _train_simulate with the current training episode
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        # prepare to display the states
        if self.parameters["display_environment"]:
            self.show_render = ShowRender()

        for t in tqdm(range(1, self.parameters["number_episodes"] + 1)):
            self._train_simulate(environment, t)

    def simulate(self, environment, seed=0):
        """
        Method used to train the RL agent.
        It calls _train_simulate method with parameter "train_episode" set to None
        :return: Nothing
        """
        # set the seeds
        np.random.seed(seed)
        environment.seed(seed)

        # prepare the file for the results
        save_results = SaveResults(self.parameters)
        save_results.write_setting()
        save_results.set_file_results_name(seed)

        # simulate
        self._train_simulate(environment)

        # write the results and write that the experiment went well
        save_results.write_reward(self.parameters["number_episodes"], self.score)
        save_results.write_message("Experiment complete.")

    @staticmethod
    def compute_total_score(o_r_d_i):
        return o_r_d_i[1]

    @staticmethod
    def check_end_agent(o_r_d_i):
        return o_r_d_i[2]

    @staticmethod
    def compute_total_reward(o_r_d_i):
        return o_r_d_i[1]

    def make_random_action(self):
        return np.random.randint(len(self.action_space))
