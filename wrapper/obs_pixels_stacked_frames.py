from collections import deque
from wrapper.obs_pixels import ObsPixelWrapper
import numpy as np
import cv2
import matplotlib.pyplot as plt


class ObsPixelStackedWrapper(ObsPixelWrapper):
    def __init__(self, env, parameters):

        super().__init__(env, parameters)
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])

    @staticmethod
    def make_gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for a2c
        return im

    def show_stacked_frames(self, stacked_image):

        image_channel_lenght = int(stacked_image.shape[2] / self.parameters["stack_images_length"])
        index = 0
        for i in range(self.parameters["stack_images_length"]):
            # for gray scale images u have to reduce the dimensions to width height for pyplot to work
            print("image number : ", i)
            image = stacked_image[:, :, index:index+image_channel_lenght]
            image = np.squeeze(image)
            plt.imshow(image)
            plt.show()
            index = index + image_channel_lenght

    def get_manager_obs(self, image):
        img_manager = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                           self.parameters["ZONE_SIZE_MANAGER_Y"])
        img_manager = ObsPixelWrapper.sample_colors(img_manager, self.parameters["THRESH_BINARY_MANAGER"])
        return img_manager

    def get_option_obs(self, image):
        img_option = ObsPixelStackedWrapper.make_gray_scale(image)
        #img_option = image / 255
        self.images_stack.append(img_option)

        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                       img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., index_image:index_image + img_option.shape[2]] = self.images_stack[i]
            index_image = index_image + img_option.shape[2]

        #self.show_stacked_frames(img_option_stacked)

        return img_option_stacked


class ObservationZoneWrapper(ObsPixelStackedWrapper):

    def __init__(self, env, parameters):
        super().__init__(env, parameters)
