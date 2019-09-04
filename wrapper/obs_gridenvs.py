from wrapper.obs_pixels_stacked_frames import ObsPixelStackedWrapper


class ObservationZoneWrapper(ObsPixelStackedWrapper):
    def get_manager_obs(self, image):
        img_manager = ObservationZoneWrapper.make_downsampled_image(image, self.parameters["ZONE_SIZE_MANAGER_X"],
                                                                    self.parameters["ZONE_SIZE_MANAGER_Y"])
        # sampling colors is often a good idea :(
        return img_manager
