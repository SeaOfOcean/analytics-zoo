import numpy as np
from scipy import misc

class Transformer(object):
    """Composes several transforms together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> Transformer([
        >>>     Crop(10, 10),
        >>>     Normalizer(0.3, 0.8)
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ChannelNormalizer(object):
    """
    Normalize image which is an numpy array by means and std of each channel. The shape of image is (height, width, channel)
    :param mean_r: mean for red channel
    :param mean_g: mean for green channel
    :param mean_b: mean for blue channel
    :param std_r: std for red channel
    :param std_g: std for green channel
    :param std_b: std for blue channel
    :return: normalized image. The shape of image is (height, width, channel)
    """
    def __init__(self, mean_r, mean_g, mean_b, std_r, std_g, std_b):
        self.mean_r = mean_r
        self.mean_g = mean_g
        self.mean_b = mean_b
        self.std_r = std_r
        self.std_g = std_g
        self.std_b = std_b

    def __call__(self, img):
        mean = np.array([self.mean_b, self.mean_g, self.mean_r])
        std = np.array([self.std_b, self.std_g, self.std_r])
        mean_sub = img[:, :] - mean
        return mean_sub[:, :] / std

class TransposeToTensor(object):
    """
    Transpose the shape of image which is an numpy array from (height, width, channel) to (channel, height, width)
    :param to_rgb: whether need to change channel from bgr to rgb
    :return: transposed image
    """
    def __init__(self, to_rgb=True):
        self.to_rgb = to_rgb

    def __call__(self, img):
        if self.to_rgb:
            return img.transpose(2, 0, 1)[(2, 1, 0), :, :]
        else:
            return img.transpose(2, 0, 1)

class Resize(object):
    """
    Resize image to specified width and height. The type of image should be uint8.
    The shape of image is (height, width, channel)
    :param resize_width: the width resized to
    :param resize_height: the height resized to
    :return: resized image.
    """
    def __init__(self, resize_width, resize_height):
        self.resize_width = resize_width
        self.resize_height = resize_height

    def __call__(self, img):
        return misc.imresize(img, (self.resize_width, self.resize_height))