import gym
import cv2


class ObsPixelWrapper(gym.ObservationWrapper):
    """
    An abstract class for a pixel based observation wrapper.
    By default, the observations of option and agent are :
    - downsampled
    - gray-scaled
    In the subclasses, we can also sample the colors
    """

    def __init__(self, env, parameters):
        super().__init__(env)
        self.parameters = parameters

    def observation(self, observation):
        image = self.get_pixels_from_obs(observation)

        # agent observation
        img_agent = self.get_agent_obs(image.copy())

        # option observation
        img_option = self.get_option_obs(image.copy())

        # render it
        return {"vanilla": image, "agent": img_agent, "option": img_option}

    def get_pixels_from_obs(self, observation):
        return observation

    @staticmethod
    def make_tuple(image):
        return tuple(tuple(tuple(color) for color in lig) for lig in image)

    @staticmethod
    def sample_colors(image, threshold):
        img = cv2.medianBlur(image, 1)
        #_, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return img

    @staticmethod
    def make_gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
        return im

    @staticmethod
    def make_downsampled_image(image, zone_size_x, zone_size_y):
        len_y = len(image)
        len_x = len(image[0])

        if (len_x % zone_size_x == 0) and (len_y % zone_size_y == 0):
            downsampling_size = (len_x // zone_size_x, len_y // zone_size_y)
            img_blurred = cv2.resize(image, downsampling_size, interpolation=cv2.INTER_AREA)
            return img_blurred

        else:
            raise Exception("The gridworld " + str(len_x) + "x" + str(len_y) +
                            " can not be fragmented into zones " + str(zone_size_x) + "x" + str(zone_size_y))

    def get_agent_obs(self, image):
        img_agent = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_AGENT_X"],
                                                           self.parameters["ZONE_SIZE_AGENT_Y"])
        img_agent = ObsPixelWrapper.make_gray_scale(img_agent)
        return img_agent

    def get_option_obs(self, image):
        img_option = ObsPixelWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_OPTION_X"],
                                                            self.parameters["ZONE_SIZE_OPTION_Y"])

        img_option = ObsPixelWrapper.make_gray_scale(img_option)
        return img_option

class ObservationZoneWrapper(ObsPixelWrapper):

    def __init__(self, env, parameters):
        super().__init__(env, parameters)
