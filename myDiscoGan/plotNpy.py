import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data_process import *
# from itertools import chain
import os
import scipy
from data_process import *
from DiscoModel import *
B0Name='/home/lix/myDiscoGan/dataset/test_npy/IMG_1.npy'
BName = '/home/lix/myDiscoGan/results/npy/0_B.npy'
ABName= '/home/lix/myDiscoGan/results/npy/0_AB.npy'
BABName='/home/lix/myDiscoGan/results/npy/0_BAB.npy'
B0 = np.load(B0Name)
B  = np.load(BName)
AB = np.load(ABName)
BAB= np.load(BABName)

# Af.transpose(1,2,0)
# imageA = torch.Tensor(imageA)
plt.subplot(221)
plt.imshow(B0)

plt.subplot(222)
plt.imshow(B)

plt.subplot(223)
plt.imshow(AB)

plt.subplot(224)
plt.imshow(BAB)

plt.show()

print(sum(sum(B)))
print('B0=',sum(sum(B0)))
print('B=', sum(sum(B)))
print('AB=',sum(sum(AB)))
print('BAB=',sum(sum(BAB)))