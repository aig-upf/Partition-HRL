from wrapper.obs_pixels_stacked_frames import ObsPixelStackedWrapper
import numpy as np


class ObservationZoneWrapper(ObsPixelStackedWrapper):
    """
    please check if this function, which renders the observation for the option, is a downgrade version of the
    parent function (in ObsPixelStackedWrapper) :

    def get_option_obs(self, image):
      img_option = ObsPixelStackedWrapper.make_gray_scale(image)
      self.images_stack.append(img_option)

      img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                     img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)
      index_image = 0
      for i in range(0, self.images_stack.__len__()):
          img_option_stacked[..., index_image:index_image + img_option.shape[2]] = self.images_stack[i]
          index_image = index_image + img_option.shape[2]

      return index_image

    1. If you like this as it is, just remove this comment
    2. If you prefer the parent function then just delete this file and in the protocols that feed the cluster, use the
    wrapper of file "obs_pixels_stacked_frames" instead

    """

    def get_option_obs(self, image):
        img_option = ObsPixelStackedWrapper.make_gray_scale(image)
        self.images_stack.append(img_option)

        img_option_stacked = np.zeros((img_option.shape[0], img_option.shape[1],
                                       img_option.shape[2] * self.parameters["stack_images_length"]), dtype=np.float32)

        for i in range(0, self.images_stack.__len__()):
            img_option_stacked[..., i - 1:i] = self.images_stack[i]
        return img_option_stacked
