import cv2
from wrapper.obs_pixels import ObsPixelWrapper


class ObservationZoneWrapper(ObsPixelWrapper):
    """
    option : downsample + gray-scale
    agent : downsample + sample_colors + hash
    """

    def __init__(self, env, parameters):

        super().__init__(env, parameters)

    def observation(self, observation):
        obs_dict = super().observation(observation)
        obs_dict["agent"] = hash(ObsPixelWrapper.make_tuple(obs_dict["agent"]))

        return obs_dict

    def get_agent_obs(self, image):
        img_agent = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_AGENT_X"],
                                                           self.parameters["ZONE_SIZE_AGENT_Y"])
        img_agent = ObsPixelWrapper.sample_colors(img_agent, self.parameters["THRESH_BINARY_AGENT"])
        return img_agent

    @staticmethod
    def make_gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for a2c
        return im
