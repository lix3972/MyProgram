from __future__ import division

import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np
import LocationCrop as Lcrop
import numbers

import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types

# class LocationCrop(object):
#     """Crops the given PIL.Image at the Location to have a region of
#     the given size. size can be a tuple (target_height, target_width)
#     or an integer, in which case the target will be of a square shape (size, size)
#     """
#
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     def __call__(self, img, x, y):
#         w, h = img.size
#         th, tw = self.size
#         x1 = min(max(x, 0), tw-x)  # int(round((w - tw) / 2.))
#         y1 = min(max(y, 0), th-y)  # int(round((h - th) / 2.))
#
#         return img.crop((x1, y1, x1 + tw, y1 + th))

A_path='1.jpg'
A = Image.open(A_path).convert('RGB')
print('A.size=', A.size)
A.show()

# transforms.ToTensor
# transforms.ToPILImage
Anp=np.array(A,dtype=np.float32)
print('Anp.shape=', Anp.shape)

Atensor=transforms.ToTensor()(A)
print('Atensor.shape=', Atensor.shape)

ACenterCrop=transforms.CenterCrop((256,256))(A)
# ACenterCrop.show()

ALocationCrop=Lcrop.LocationCrop((256,256))(A, 400, 400)
# ALocationCrop.show()
print('ALocationCrop.size=', ALocationCrop.size,'mode=',ALocationCrop.mode )

Acrop = A.crop((0,0,256,256))
print(Acrop.size)
Acrop.show()