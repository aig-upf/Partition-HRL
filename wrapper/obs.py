__all__ = ["PixelsStackedFrames", "Minigrid", "A2C", "GridWorld", "Cluster", "GrayManager"]

from abstract.utils.miscellaneous import make_tuple
from collections import deque
from abstract.utils.observation_wrapper import ObsPixelWrapper
import numpy as np
import cv2
import matplotlib.pyplot as plt


class A2C(ObsPixelWrapper):
    """
    option : downsample + gray-scale
    manager : downsample + sample_colors + hash
    """

    def __init__(self, env, parameters):

        super().__init__(env, parameters)

    def observation(self, observation):
        obs_dict = super().observation(observation)
        obs_dict["manager"] = hash(make_tuple(obs_dict["manager"]))

        return obs_dict

    def get_manager_obs(self, image):
        img_manager = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                             self.parameters["ZONE_SIZE_MANAGER_Y"])
        img_manager = ObsPixelWrapper.sample_colors(img_manager, self.parameters["THRESH_BINARY_MANAGER"])
        return img_manager

    @staticmethod
    def make_gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for manager
        return im


class PixelsStackedFrames(ObsPixelWrapper):
    def __init__(self, env, parameters):

        super().__init__(env, parameters)
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])

    @staticmethod
    def make_gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for manager
        return im

    def show_stacked_frames(self, stacked_image):

        image_channel_length = int(stacked_image.shape[2] / self.parameters["stack_images_length"])
        index = 0
        for i in range(self.parameters["stack_images_length"]):
            # for gray scale images u have to reduce the dimensions to width height for pyplot to work
            print("image number : ", i)
            image = stacked_image[:, :, index:index + image_channel_length]
            image = np.squeeze(image)
            plt.imshow(image)
            plt.show()
            index = index + image_channel_length

    def get_manager_obs(self, image):

        #img_manager = ObsPixelWrapper.make_gray_scale(image)

        img_manager = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                             self.parameters["ZONE_SIZE_MANAGER_Y"])
        img_manager = ObsPixelWrapper.sample_colors(img_manager, self.parameters["THRESH_BINARY_MANAGER"])

        #img_manager = np.concatenate((img_manager, img_manager, img_manager))
        #print(img_manager.shape)
        return img_manager

    def get_option_obs(self, image):
        img_option = self.make_gray_scale(image)
        # img_option = image / 255
        self.images_stack.append(img_option)

        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                       img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., index_image:index_image + img_option.shape[2]] = self.images_stack[i]
            index_image = index_image + img_option.shape[2]

        # self.show_stacked_frames(img_option_stacked)

        return img_option_stacked


class Minigrid(PixelsStackedFrames):
    """
    same observation as ObsPixelStackedWrapper
    """
    def get_pixels_from_obs(self, observation):
        return self.env.render(mode='rgb_array', highlight=False)  # to get the rgb image


class Gridenvs(PixelsStackedFrames):
    def get_manager_obs(self, image):
        img_manager = self.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                  self.parameters["ZONE_SIZE_MANAGER_Y"])
        # sampling colors is often a good idea :(
        return img_manager


class GridWorld(ObsPixelWrapper):
    """
    Same as ObsPixelWrapper but the option observation is not transformed
    """

    def get_option_obs(self, image):
        return image


class Cluster(PixelsStackedFrames):

    def get_option_obs(self, image):
        img_option = PixelsStackedFrames.make_gray_scale(image)
        self.images_stack.append(img_option)

        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                       img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)

        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., i - 1:i] = self.images_stack[i]
        return img_option_stacked


class GrayManager(PixelsStackedFrames):

    def get_manager_obs(self, image):
        img_manager = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                             self.parameters["ZONE_SIZE_MANAGER_Y"])
        img_manager = ObsPixelWrapper.make_gray_scale(img_manager)
        # _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY) ?
        return img_manager
