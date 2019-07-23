from wrapper.obs_pixels_stacked_frames import ObsPixelStackedWrapper


class ObservationZoneWrapper(ObsPixelStackedWrapper):
    def get_agent_obs(self, image):
        img_agent = ObservationZoneWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_AGENT_X"],
                                                           self.parameters["ZONE_SIZE_AGENT_Y"])
        return img_agent
