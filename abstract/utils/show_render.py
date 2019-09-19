import cv2
import pyglet
import numpy as np


class ShowRender(object):

    def __init__(self):
        from gym.envs.classic_control import rendering  # causes problem with the cluster

        self.display_learning = True

        self.vanilla_view = True
        self.option_view = False
        self.agent_view = False

        self.viewer = rendering.SimpleImageViewer()
        self.viewer.width = 512
        self.viewer.height = 512
        self.viewer.window = pyglet.window.Window(width=self.viewer.width, height=self.viewer.height,
                                                  display=self.viewer.display, vsync=False, resizable=True)
        self.viewer.window.on_key_press = self.key_press

    def render(self, observation):
        """
        :param observation: a dictionary containing the observations:
        - observation vanilla
        - observation manager
        - observation option
        :return:
        """
        if self.display_learning:
            if type(observation).__name__ == "ndarray":
                self.display(observation)

            else:
                assert list(observation.keys()) == ["vanilla", "manager", "option"], \
                    "observation must be a dictionary with 3 keys : vanilla, manager and option"

                if self.vanilla_view:
                    self.display(observation["vanilla"])

                elif self.agent_view:
                    self.display(observation["manager"])

                elif self.option_view:
                    self.display(observation["option"])

        else:
            self.viewer.imshow(np.array([[[0]]]))

    def display(self, image_pixel):
        img = cv2.resize(image_pixel, (512, 512), interpolation=cv2.INTER_NEAREST)
        self.viewer.imshow(img)

    def close(self):
        self.viewer.close()

    def key_press(self, key, mod):
        if key == ord("d"):
            print("press d to display the observation")
            self.display_learning = not self.display_learning

        if key == ord("o"):
            self.set_option_view()

        if key == ord("v"):
            self.set_vanilla_view()

        if key == ord("a"):
            self.set_agent_view()

        if key == ord(" "):
            if self.option_view:
                self.set_vanilla_view()

            elif self.vanilla_view:
                self.set_agent_view()

            elif self.agent_view:
                self.set_option_view()

    def set_vanilla_view(self):
        print("original view")
        self.vanilla_view = True
        self.option_view = False
        self.agent_view = False

    def set_option_view(self):
        print("option's view")
        self.vanilla_view = False
        self.option_view = True
        self.agent_view = False

    def set_agent_view(self):
        print("manager's view")
        self.vanilla_view = False
        self.option_view = False
        self.agent_view = True
