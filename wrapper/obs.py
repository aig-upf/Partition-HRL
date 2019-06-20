import gym
import cv2
from gym.envs.classic_control import rendering


class ObservationZoneWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env,
                 zone_size_option_x,
                 zone_size_option_y,
                 zone_size_agent_x,
                 zone_size_agent_y,
                 thresh_binary_option,
                 thresh_binary_agent):

        super().__init__(env)
        self.zone_size_option_x = zone_size_option_x
        self.zone_size_option_y = zone_size_option_y
        self.zone_size_agent_x = zone_size_agent_x
        self.zone_size_agent_y = zone_size_agent_y
        self.thresh_binary_option = thresh_binary_option
        self.thresh_binary_agent = thresh_binary_agent

    def render(self,
               size=(512, 512),
               agent_render=True,
               close=False,
               blurred_render=False,
               gray_scale_render=False):

        if hasattr(self.env.__class__, 'render_scaled'):  # we call render_scaled function from gridenvs
            return self.env.render_scaled(size, "human", close)
         
        else:  # we scale the image from other environment (like Atari)
            env_unwrapped = self.env.unwrapped
            img = env_unwrapped.ale.getScreenRGB2()

            if blurred_render:
                if agent_render:
                    img = ObservationZoneWrapper.make_downsampled_image(img,
                                                                        self.zone_size_agent_x,
                                                                        self.zone_size_agent_y)
                else:
                    img = ObservationZoneWrapper.make_downsampled_image(img,
                                                                        self.zone_size_option_x,
                                                                        self.zone_size_option_y)

            if gray_scale_render:
                if agent_render:
                    img = ObservationZoneWrapper.make_gray_scale(img, self.thresh_binary_agent)
                else:
                    img = ObservationZoneWrapper.make_gray_scale(img, self.thresh_binary_option)

            img_resize = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

            if env_unwrapped.viewer is None:
                env_unwrapped.viewer = rendering.SimpleImageViewer()

            env_unwrapped.viewer.imshow(img_resize)
            return env_unwrapped.viewer.isopen

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

    def observation(self, observation):
        img_option = observation
        img_agent = img_option.copy()

        img_option = ObservationZoneWrapper.make_downsampled_image(img_option,
                                                                   self.zone_size_option_x,
                                                                   self.zone_size_option_y)

        img_option = ObservationZoneWrapper.make_gray_scale(img_option,
                                                            self.thresh_binary_option)

        img_agent = ObservationZoneWrapper.make_downsampled_image(img_agent,
                                                                  self.zone_size_agent_x,
                                                                  self.zone_size_agent_y)

        img_agent = ObservationZoneWrapper.make_gray_scale(img_agent, self.thresh_binary_agent)

        img_option_tuple = tuple(tuple(tuple(color) for color in lig) for lig in img_option)
        img_agent_tuple = tuple(tuple(tuple(color) for color in lig) for lig in img_agent)

        return {"agent": hash(img_agent_tuple), "option": img_option}

    @staticmethod
    def make_gray_scale(image, threshold):
        img = cv2.medianBlur(image, 1)
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return img
