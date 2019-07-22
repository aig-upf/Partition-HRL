from wrapper.obs_pixels_stacked_frames import ObsPixelStackedWrapper

class ObservationZoneWrapper(ObsPixelStackedWrapper):
    """
    same observation as ObsPixelStackedWrapper
    """
    def get_pixels_from_obs(self, observation):
        return self.env.render(mode='rgb_array', highlight=False)  # to get the rgb image
