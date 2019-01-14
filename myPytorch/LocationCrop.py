from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types

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
        th, tw = self.size
        x1 = min(max(x, 0), w-tw)  # int(round((w - tw) / 2.))
        y1 = min(max(y, 0), h-th)  # int(round((h - th) / 2.))

        return img.crop((x1, y1, x1 + tw, y1 + th))