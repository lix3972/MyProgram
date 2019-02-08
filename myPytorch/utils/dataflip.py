from __future__ import division
import torch
import math
import random


def hflip_T(data_tensor):
    c, h, w = data_tensor.shape  # chanel, hight, width
    tmp = torch.empty(c, h, w)
    for i in range(c):
        for j in range(h):
            for k in range(w):
                tmp[i, j, k] = data_tensor[i, j, w-1-k]
    return tmp

class RandomHorizontalFlip_T(object):
    """Horizontally flip the given Tensor randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_Tensor):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return hflip_T(data_Tensor)
        return data_Tensor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
