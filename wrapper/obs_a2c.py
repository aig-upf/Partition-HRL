import cv2
from wrapper.obs_pixels import ObsPixelWrapper


class ObservationZoneWrapper(ObsPixelWrapper):
    """
    option : downsample + gray-scale
    manager : downsample + sample_colors + hash
    """

    def __init__(self, env, parameters):

        super().__init__(env, parameters)

    def observation(self, observation):
        obs_dict = super().observation(observation)
        obs_dict["manager"] = hash(ObsPixelWrapper.make_tuple(obs_dict["manager"]))

        return obs_dict

    def get_manager_obs(self, image):
        img_manager = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                           self.parameters["ZONE_SIZE_MANAGER_Y"])
        img_manager = ObsPixelWrapper.sample_colors(img_manager, self.parameters["THRESH_BINARY_MANAGER"])
        return img_manager

    @staticmethod
    def make_gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for a2c
        return im
