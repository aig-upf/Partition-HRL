from collections import deque
from abstract.utils.observation_wrapper import ObsPixelWrapper
import numpy as np
import cv2
import matplotlib.pyplot as plt


class ObsPixelStackedWrapper(ObsPixelWrapper):
    def __init__(self, env, parameters):

        super().__init__(env, parameters)
        self.images_stack = deque([], maxlen=self.parameters["stack_images_length"])
        self.old_abstract_state = 0
        self.abstract_states = []

    @staticmethod
    def make_gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #im = im.reshape((*im.shape, 1))  # reshape for the neural network for manager
        return im

    @staticmethod
    def normalize(image):
        im = image / 255
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

    def check_SSIM_abstract_state(self, abstract_state):

        if not self.abstract_states:
            self.abstract_states.append(abstract_state)
            print(" Abstract state - ",
                  [ObsPixelWrapper.SSIM_equal(abstract_state, x, not self.parameters["GRAY_SCALE"], True)
                   for x in self.abstract_states].index(True),
                  )
            self.old_abstract_state = abstract_state

            return

        if ObsPixelWrapper.ssim_in_list(abstract_state, self.abstract_states, not self.parameters["GRAY_SCALE"]):
            if ObsPixelWrapper.SSIM_equal(abstract_state, self.old_abstract_state, not self.parameters["GRAY_SCALE"]):
                return
            else:
                print(" Abstract state - ",
                      [ObsPixelWrapper.SSIM_equal(abstract_state, x, not self.parameters["GRAY_SCALE"], True)
                       for x in self.abstract_states].index(True))

                self.old_abstract_state = abstract_state

        else:
            self.abstract_states.append(abstract_state)
            print(" New Abstract state - ",
                  [ObsPixelWrapper.SSIM_equal(abstract_state, x, not self.parameters["GRAY_SCALE"], True)
                   for x in self.abstract_states].index(True))


    def get_manager_obs(self, image):

        if self.parameters["GRAY_SCALE"]:
            image = ObsPixelStackedWrapper.make_gray_scale(image)

        #img_manager = ObsPixelWrapper.sample_colors(img_manager, self.parameters["THRESH_BINARY_MANAGER"])

        img_manager = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                             self.parameters["ZONE_SIZE_MANAGER_Y"])

        #self.check_SSIM_abstract_state(img_manager)

        return img_manager

    def get_option_obs(self, image):
        img_option = ObsPixelStackedWrapper.normalize(image)
        self.images_stack.append(img_option)

        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                       img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)
        index_image = 0
        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., index_image:index_image + img_option.shape[2]] = self.images_stack[i]
            index_image = index_image + img_option.shape[2]

        # self.show_stacked_frames(img_option_stacked)

        return img_option_stacked


class ObservationZoneWrapper(ObsPixelStackedWrapper):

    def __init__(self, env, parameters):
        super().__init__(env, parameters)
