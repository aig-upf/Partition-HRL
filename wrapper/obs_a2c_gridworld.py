from wrapper.obs_pixels import ObsPixelWrapper


class ObservationZoneWrapper(ObsPixelWrapper):
    """
    Same as ObsPixelWrapper but the option observation is not transformed
    """
    def get_option_obs(self, image):
        return image
