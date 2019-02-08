
import torch
# import math
# import random
# from PIL import Image, ImageOps
# import numpy as np
import numbers
# import types

class LocationCrop(object):
    """Crops the given PIL.Image at the Location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, x, y):
        w, h = img.size
        tw, th = self.size
        x1 = min(max(x, 0), w-tw)  # int(round((w - tw) / 2.))
        y1 = min(max(y, 0), h-th)  # int(round((h - th) / 2.))

        return img.crop((x1, y1, x1 + tw, y1 + th))

class LocationCropNpy(object):
    """Crops the given PIL.Image at the Location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, x, y):
        _, w, h = img.shape
        tw, th = self.size
        x1 = min(max(x, 0), w-tw)  # int(round((w - tw) / 2.))
        y1 = min(max(y, 0), h-th)  # int(round((h - th) / 2.))
        return img[:, x1:x1 + tw, y1:y1 + th]

def cropTensorTo3_3(im,image_size=768):
    w = int(image_size / 3)
    h = int(image_size / 3)

    im1 = im[:, 0:h, 0:w]
    im2 = im[:, h:2*h, 0:w]
    im3 = im[:, 2*h:3*h, 0:w]
    im4 = im[:, 0:h, w:2*w]
    im5 = im[:, h:2*h, w:2*w]
    im6 = im[:, 2*h:3*h, w:2*w]
    im7 = im[:, 0:h, 2*w:3*w]
    im8 = im[:, h:2*h, 2*w:3*w]
    im9 = im[:, 2*h:3*h, 2*w:3*w]

    fullim = torch.stack([im1, im2, im3, im4, im5, im6, im7, im8, im9], 0)
    return fullim

def cropTensorTo2_2(im,image_size=256):
    w = int(image_size / 2)
    h = int(image_size / 2)

    im1 = im[:, :, 0:h, 0:w]  # [batch_size, chanel, h, w]
    im2 = im[:, :, h:2*h, 0:w]
    im3 = im[:, :, 0:h, w:2*w]
    im4 = im[:, :, h:2*h, w:2*w]

    fullim = torch.cat([im1, im2, im3, im4], 0)
    return fullim
