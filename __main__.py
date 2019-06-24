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
from docopt import docopt
import importlib.util
# todo add the right wrapper following the protocol information (a2c wrapper or regular wrapper)
import tensorflow as tf
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.enable_eager_execution()
# todo fix this The name tf.enable_eager_execution is deprecated. Please use tf.compat.v1.enable_eager_execution instead


# Just to be sure that we don't have some others graph loaded
tf.reset_default_graph()
# todo fix this:  The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.


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
        print("charging the environment: " + str(self.parameters["env_name"]))
        time.sleep(0.5)
        env = gym.make(self.parameters["env_name"]).env

        if "obs_wrapper_name" in self.parameters.keys():
            print("observation wrapper name is " + str(self.parameters["obs_wrapper_name"]))
            time.sleep(0.5)

            if self.parameters["obs_wrapper_name"] in ["obs", "obs_a2c"] :
                obs = getattr(importlib.import_module("wrapper." + self.parameters["obs_wrapper_name"]),
                              "ObservationZoneWrapper")

                return obs(env,
                           zone_size_option_x=self.parameters["ZONE_SIZE_OPTION_X"],
                           zone_size_option_y=self.parameters["ZONE_SIZE_OPTION_Y"],
                           zone_size_agent_x=self.parameters["ZONE_SIZE_AGENT_X"],
                           zone_size_agent_y=self.parameters["ZONE_SIZE_AGENT_Y"],
                           thresh_binary_option=self.parameters["THRESH_BINARY_OPTION"],
                           thresh_binary_agent=self.parameters["THRESH_BINARY_AGENT"])
            else:
                raise Exception("wrapper name unknown.")

        else:
            print("No observation wrapper.")
            time.sleep(0.5)
            return env

    def get_agent(self):
        """
        :return: the agent with parameters specified in the parameters
        """
        print("agent : " + str(self.parameters["agent_name"]))
        time.sleep(1.5)
        agent = getattr(importlib.import_module("agent." + self.parameters["agent_file"]), self.parameters["agent_name"])
        return agent(action_space=range(self.env.action_space.n), parameters=self.parameters)

    def run(self):
        # loop on the seed to simulate the agent
        for seed in self.parameters["seeds"]:

            # first, train the agent
            self.agent.train_agent(self.env, seed)

            # wait for the signal to run the simulation
            input("PRESS ANY KEY")

            # set the simulate environment and test the agent
            self.agent.simulate_agent(self.env, seed)


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
