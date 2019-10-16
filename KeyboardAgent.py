import sys
from abstract.utils.show_render import ShowRender
import pyglet
import importlib.util
import gym
import time
import gridenvs.examples
import gym_minigrid


class ShowRenderKeyboard(ShowRender):

    def __init__(self, number_of_actions, parameters):
        from gym.envs.classic_control import rendering  # causes problem with the cluster

        self.display_learning = True

        self.vanilla_view = False
        self.option_view = False
        self.agent_view = False
        self.combined_view = True
        self.slow_display = 0

        self.ACTIONS = number_of_actions
        self.list_of_actions_keys = parameters["Keyboard_keys"][0:number_of_actions]
        print(self.list_of_actions_keys)

        self.viewer = rendering.SimpleImageViewer()
        self.viewer.width = 512
        self.viewer.height = 512
        self.viewer.window = pyglet.window.Window(width=self.viewer.width, height=self.viewer.height,
                                                  display=self.viewer.display, vsync=False, resizable=True)

        self.viewer.window.on_key_press = self.key_press
        self.viewer.window.on_key_release = self.key_release

        self.SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
        # can test what skip is still usable.

        self.human_agent_action = 0
        self.human_wants_restart = False
        self.human_sets_pause = False

        self.Print=True

    def key_press(self, key, mod):
        if key == 0xff0d: self.human_wants_restart = True
        #if key == 32: self.human_sets_pause = not self.human_sets_pause

        if key == ord(" "):
            if self.option_view:
                self.set_combined_view()

            elif self.vanilla_view:
                self.set_agent_view()

            elif self.agent_view:
                self.set_option_view()

            elif self.combined_view:
                self.set_vanilla_view()

        if key in self.list_of_actions_keys:
            a = int(self.list_of_actions_keys.index(key))
            self.human_agent_action = a

    def key_release(self, key, mod):
        a = 0
        if key in self.list_of_actions_keys:
            a = int(self.list_of_actions_keys.index(key))
            self.human_agent_action = a
        if self.human_agent_action == a:
            self.human_agent_action = 0


class Experiment(object):
    """
    This class makes an experiment and an manager from a protocol
    """

    def __init__(self, protocol_exp):
        # the manager and environment's parameters are set in the protocol
        self.parameters = protocol_exp
        self.results_paths = []
        self.env = self.get_environment()
        self.show_render = self.get_show_render(self.env.action_space.n, parameters)
        self.total_reward = 0
        self.total_timesteps = 0

        print("ACTIONS={}".format(self.show_render.ACTIONS))
        print("Press keys ... to take actions ...")
        print("No keys pressed is taking action 0")

    @staticmethod
    def get_show_render(number_of_actions, parameters):
        return ShowRenderKeyboard(number_of_actions, parameters)

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

    def run(self):
        o_r_d_i = [self.env.reset()] + [None]*3
        self.show_render.render(o_r_d_i[0])
        self.show_render.human_wants_restart = False

        while 1:
            a = self.show_render.human_agent_action
            self.total_timesteps += 1
            o_r_d_i = self.env.step(a)
            if o_r_d_i[1] != 0:
                print("reward %0.3f" % o_r_d_i[1])
                self.total_reward += o_r_d_i[1]

            self.show_render.render(o_r_d_i[0])

            if o_r_d_i[2]:
                print("DONE")
                time.sleep(1)
                break

            if self.show_render.human_wants_restart: break

            while self.show_render.human_sets_pause:
                self.env.render()
                time.sleep(0.1)
            time.sleep(0.1)
        print("timesteps %i total reward %0.2f" % (self.total_timesteps, self.total_reward))
        self.total_reward = 0


if __name__ == '__main__':
    # Parse command line arguments
    args = sys.argv[1]

    # Get the protocol info
    path_protocol = 'protocols.' + args
    parameters = importlib.import_module(path_protocol).data
    parameters["path"] = path_protocol

    # Create an experiment
    experiment = Experiment(parameters)

    # Run the experiment : train and simulate the manager and store the results
    while 1:
        experiment.run()
