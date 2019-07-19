import gym
import cv2
from collections import deque
import numpy as np


class ObservationZoneWrapper(gym.ObservationWrapper):
    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.zone_size_agent_x = self.parameters["ZONE_SIZE_AGENT_X"]
        self.zone_size_agent_y = self.parameters["ZONE_SIZE_AGENT_Y"]
        self.thresh_binary_option = self.parameters["THRESH_BINARY_OPTION"]
        self.thresh_binary_agent = self.parameters["THRESH_BINARY_AGENT"]
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])

        self.observation_agent = None
        self.observation_option = None

    def observation(self, observation):
        vanilla = self.env.render(mode='rgb_array', highlight=False)  # to get the rgb image
        img_option = vanilla.copy()
        img_agent = vanilla.copy()

        # option observation
        img_option = ObservationZoneWrapper.gray_scale(img_option)
        self.images_stack.append(img_option)
        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                img_option.shape[2]*self.parameters["stack_images_length"]), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., index_image:index_image+img_option.shape[2]] = self.images_stack[i]
            index_image = index_image+img_option.shape[2]

        # agent observation
        img_agent = ObservationZoneWrapper.make_downsampled_image(img_agent, self.zone_size_agent_x,
                                                                  self.zone_size_agent_y)
        img_agent = ObservationZoneWrapper.sample_colors(img_agent, self.thresh_binary_agent)

        # store the observations
        self.observation_agent = img_agent
        self.observation_option = img_option

        return {'vanilla': vanilla, "agent": img_agent, "option": img_option_stacked}

    @staticmethod
    def sample_colors(image, threshold):
        img = cv2.medianBlur(image, 1)
        #_, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return img

    @staticmethod
    def gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for a2c
        return im

    @staticmethod
    def make_downsampled_image(image, zone_size_x, zone_size_y):
        len_y = len(image)  # with MontezumaRevenge-v4 : 160
        len_x = len(image[0])  # with MontezumaRevenge-v4 : 210
        if (len_x % zone_size_x == 0) and (len_y % zone_size_y == 0):
            downsampling_size = (len_x // zone_size_x, len_y // zone_size_y)
            # vector of size "downsampled_size"
            img_blurred = cv2.resize(image, downsampling_size, interpolation=cv2.INTER_AREA)
            return img_blurred

        else:
            raise Exception("The gridworld " + str(len_x) + "x" + str(len_y) +
                            " can not be fragmented into zones " + str(zone_size_x) + "x" + str(zone_size_y))
