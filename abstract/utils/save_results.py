import os
import time
import matplotlib.pyplot as plt


class SaveResults(object):

    def __init__(self):
        self.dir_path = self.make_dir_path()
        self.dir_path_seed = None

        self.manager_score_file_name = "manager_score"
        self.success_rate_file_name = "success_rate_transition"

    def set_seed(self, seed):
        self.dir_path_seed = self.dir_path + "/seed_" + str(seed)
        os.mkdir(self.dir_path_seed)

    @staticmethod
    def make_dir_path():
        dir_name = "results/"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        dir_name += time.asctime(time.localtime(time.time())).replace(" ", "_")
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        return dir_name

    def write_message_in_a_file(self, file_name, message):
        f = open(self.dir_path_seed + "/" + file_name, "a")
        f.write(message)
        f.close()

    def write_setting(self, parameters):
        """
        writes the parameters in a file
        """
        f = open(self.dir_path + "/" + "setting", "w")
        for key in parameters:
            f.write(key + " : " + str(parameters[key]) + "\n")

        f.close()

    def plot_results(self, file_name, title, xlabel, ylabel):
        x, y = [], []
        with open(str(self.dir_path_seed) + "/" + file_name) as f:
            for line in f:
                x.append(float(line.split()[0]))
                y.append(float(line.split()[1]))

        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.draw()
        # plt.pause(0.01)
        plt.savefig(str(self.dir_path_seed) + "/" + file_name)
        plt.savefig("metrics/" + file_name)
        plt.close()

    def plot_success_rate_transitions(self):
        self.plot_results(self.success_rate_file_name, "success rate of options' transitions",
                          "number of options executed", "% of successful option executions")

    def plot_manager_score(self):
        self.plot_results(self.manager_score_file_name, "manager's average score",
                          "epochs", "average total reward in epochs")
