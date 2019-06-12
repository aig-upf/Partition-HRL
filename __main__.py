# -*- coding: utf-8 -*-
""" Delta project
 Creates an agent which interacts with the Montezuma's revenge

Usage:
    Montezuma_RL [options] <protocol>

where
    <protocol> is the name of the Python file describing the parameters of the agent.
    The protocol file is available in protocols/<protocol>.py
    The parameters set in this file can be overwritten by the options below, specified in the command line.

Options:
    -h                          Display this help.
    -o PATH                     Output path.
"""

import gym
import os
from docopt import docopt
import importlib.util
from shutil import copyfile
from datetime import datetime
from wrapper.obs import ObservationZoneWrapper
from agent.agent import Agent


class Experiment(object):
    """
    This class makes an experiment and an agent from a protocol
    """

    def __init__(self, protocol_exp):
        # the agent and environment's parameters are set in the protocol
        self.parameters = protocol_exp
        self.env = self.get_environment()
        self.agent = self.get_agent()

    def get_environment(self):
        """
        :return: the environment with parameters specified in the protocol
        """
        env = gym.make("MontezumaRevenge-v0").env
        return ObservationZoneWrapper(env,
                                      zone_size_option_x=self.parameters["ZONE_SIZE_OPTION_X"],
                                      zone_size_option_y=self.parameters["ZONE_SIZE_OPTION_Y"],
                                      zone_size_agent_x=self.parameters["ZONE_SIZE_AGENT_X"],
                                      zone_size_agent_y=self.parameters["ZONE_SIZE_AGENT_Y"],
                                      thresh_binary_option=self.parameters["THRESH_BINARY_OPTION"],
                                      thresh_binary_agent=self.parameters["THRESH_BINARY_AGENT"])

    def get_agent(self):
        """
        :return: the agent with parameters specified in the parameters
        """
        return Agent(action_space=range(self.env.action_space.n), parameters=self.parameters)

    def run(self):
        # Create results/ folder (if needed)
        results_folder = "results/results_%s_%s" % ("Montezuma", datetime.now().strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(results_folder)

        # loop on the seed to simulate the agent
        for seed in self.parameters["seeds"]:
            self.env.seed(seed)

            # first, train the agent
            self.agent.train_agent(self.env, seed)

            # set the simulate environment and test the agent
            self.agent.simulate_agent(self.env, seed)

        # copy the protocol file in the results file
        copyfile(self.parameters["path"], results_folder + "/protocol.py")


if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)

    # Get the protocol info
    path_protocol = 'protocols.' + args['<protocol>']
    parameters = importlib.import_module(path_protocol).data
    parameters["path"] = path_protocol

    # Create an experiment
    experiment = Experiment(parameters)

    # Run the experiment : train and simulate the agent and store the results
    experiment.run()