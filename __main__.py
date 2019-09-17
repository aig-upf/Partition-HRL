# -*- coding: utf-8 -*-
""" Delta project
 Creates an manager which interacts with the Montezuma's revenge

Usage:
    Montezuma_RL [options] <protocol>

where
    <protocol> is the name of the Python file describing the parameters of the manager.
    The protocol file is available in protocols/<protocol>.py
    The parameters set in this file can be overwritten by the options below, specified in the command line.

Options:
    -h                          Display this help.
    -o PATH                     Output path.
"""

import gym
import gridenvs.examples
import gym_minigrid

from docopt import docopt
import importlib.util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Experiment(object):
    """
    This class makes an experiment and an manager from a protocol
    """

    def __init__(self, protocol_exp):
        # the manager and environment's parameters are set in the protocol
        self.parameters = protocol_exp
        self.results_paths = []
        self.env = self.get_environment()
        self.manager = self.get_manager()

    def get_environment(self):
        """
        :return: the environment with parameters specified in the protocol
        """
        print("charging the environment: " + str(self.parameters["env_name"]))
        env = gym.make(self.parameters["env_name"])

        if "obs_wrapper_name" in self.parameters.keys():
            print("observation wrapper name is " + str(self.parameters["obs_wrapper_name"]))
            obs = getattr(importlib.import_module("wrapper." + self.parameters["obs_wrapper_name"]),
                          "ObservationZoneWrapper")

            return obs(env, self.parameters)

        else:
            print("No observation wrapper.")
            return env

    def get_manager(self):
        """
        :return: the manager with parameters specified in the parameters
        """
        print("manager : " + str(self.parameters["manager_name"]))
        m = getattr(importlib.import_module(self.parameters["manager_file"]),
                    self.parameters["manager_name"])
        return m(action_space=range(self.env.action_space.n), parameters=self.parameters)

    def run(self):
        # loop on the seed to simulate the manager
        for seed in self.parameters["seeds"]:

            # first, train the manager
            self.manager.reset_all()
            self.manager.train(self.env, self.parameters, seed)
            self.results_paths.append(self.manager.get_result_paths())

        print("Learning phase: Done.")
        # wait for the signal to run the simulation
        # input("Learning phase: Done. Press any key to run the simulation")

        # set the simulate environment and test the manager
        # self.manager.simulate(self.env, seed)

    def plot(self, plot_type, title, xlabel, ylabel):

        list_results = [np.array(pd.read_csv(p[plot_type], header=None)) for p in self.results_paths]
        results_length = [len(result) for result in list_results]
        min_length = min(results_length)

        if len(set(results_length)) != 1:
            # arrays have different size
            # slice arrays
            list_results = [a[:min_length] for a in list_results]

        st = np.hstack(list_results)

        x = range(min_length)
        y_mean = np.mean(st, axis=-1)
        y_max = np.max(st, axis=-1)
        y_min = np.min(st, axis=-1)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.plot(x, y_mean, color='#CC4F1B')
        plt.fill_between(x, y_min, y_max, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.savefig(self.manager.get_result_folder() + "/" + plot_type)

        plt.close()


if __name__ == '__main__':
    # Parse command line arguments
    args = docopt(__doc__)

    # Get the protocol info
    path_protocol = 'protocols.' + args['<protocol>']
    parameters = importlib.import_module(path_protocol).data
    parameters["path"] = path_protocol

    # Create an experiment
    experiment = Experiment(parameters)

    # Run the experiment : train and simulate the manager and store the results
    experiment.run()

    # Plot results
    experiment.plot("manager", title="manager's score", xlabel="epochs", ylabel="total reward in epochs")
    experiment.plot("transitions", title="success rate of options' transitions", xlabel="number of options executed",
                    ylabel="% of successful option executions")
