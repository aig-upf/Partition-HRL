import numpy as np
from collections import deque


class ExperienceReplay:

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, random=True):
        buffer_size = len(self.buffer)
        if random:
            index = np.random.choice(np.arange(buffer_size),
                                     size=batch_size,
                                     replace=False)
        else:
            index = np.arange(buffer_size)

        sample = [(i, self.buffer[i]) for i in index]

        return sample

    def buffer_len(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer.clear()


class Preprocessing:

    def __init__(self, image_width, image_height, image_depth):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth

    def preprocess_image(self, img):
        #img = scipy.misc.imresize(img, ( self.image_width, self.image_height), interp='nearest')

        img = img / 255
        return img