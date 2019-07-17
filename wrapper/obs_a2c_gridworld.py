import gym
import cv2
from gym.envs.classic_control import rendering


class ObservationZoneWrapper(gym.ObservationWrapper):
    def __init__(self, env, parameters):

        super().__init__(env)
        self.parameters = parameters
        self.zone_size_agent_x = self.parameters["ZONE_SIZE_AGENT_X"]
        self.zone_size_agent_y = self.parameters["ZONE_SIZE_AGENT_Y"]
        self.thresh_binary_agent = self.parameters["THRESH_BINARY_AGENT"]

    def render(self, size=(512, 512), agent_render=True, close=False, blurred_render=False, gray_scale_render=False):

        if hasattr(self.env.__class__, 'render_scaled'):  # we call render_scaled function from gridenvs
            return self.env.render_scaled(size, "human", close)
         
        else:  # we scale the image from other environment (like Atari)
            env_unwrapped = self.env.unwrapped
            img = env_unwrapped.ale.getScreenRGB2()

            if blurred_render:
                if agent_render:
                    img = ObservationZoneWrapper.make_downsampled_image(img, self.zone_size_agent_x,
                                                                        self.zone_size_agent_y)
                else:
                    img = ObservationZoneWrapper.make_downsampled_image(img, self.zone_size_option_x,
                                                                        self.zone_size_option_y)

            if gray_scale_render:
                if agent_render:
                    img = ObservationZoneWrapper.sample_colors(img, self.thresh_binary_agent)
                else:
                    img = ObservationZoneWrapper.sample_colors(img, self.thresh_binary_option)

            img_resize = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

            if env_unwrapped.viewer is None:
                env_unwrapped.viewer = rendering.SimpleImageViewer()

            env_unwrapped.viewer.imshow(img_resize)
            return env_unwrapped.viewer.isopen

    @staticmethod
    def make_downsampled_image(image, zone_size_x, zone_size_y):
        len_y = len(image)  # with gridworld: 84
        len_x = len(image[0])  # with gridworld: 84
        if (len_x % zone_size_x == 0) and (len_y % zone_size_y == 0):
            downsampling_size = (len_x // zone_size_x, len_y // zone_size_y)
            # vector of size "downsampled_size"
            img_blurred = cv2.resize(image, downsampling_size, interpolation=cv2.INTER_AREA)
            return img_blurred

        else:
            raise Exception("The gridworld " + str(len_x) + "x" + str(len_y) +
                            " can not be fragmented into zones " + str(zone_size_x) + "x" + str(zone_size_y))

    def observation(self, observation):
        img_option = observation
        img_agent = img_option.copy()

        # agent observation
        img_agent = ObservationZoneWrapper.make_downsampled_image(img_agent,
                                                                  self.zone_size_agent_x,
                                                                  self.zone_size_agent_y)

        img_agent = ObservationZoneWrapper.sample_colors(img_agent, self.thresh_binary_agent)
        img_agent_tuple = tuple(tuple(tuple(color) for color in lig) for lig in img_agent)

        return {"agent": hash(img_agent_tuple), "option": img_option}

    @staticmethod
    def sample_colors(image, threshold):
        img = cv2.medianBlur(image, 1)
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return img

    @staticmethod
    def gray_scale(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im = im.reshape((*im.shape, 1))  # reshape for the neural network for a2c
        return im