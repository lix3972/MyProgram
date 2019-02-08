import torch
import torch.nn as nn
import argparse
import os
import time
from itertools import chain
from matplotlib import pyplot as plt
import random
import scipy
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

import data.LocationCrop as Lcrop
# from data.crowd_dataset import CrowdDataset
from models.acscp_model import *
# from models.losses import *
from data.LocationCrop import *
from utils.loadVgg import loadVgg2
from data.image_folder import make_dataset, get_npy_filenames, read_npy
from utils.dataflip import *

# 行=高=h=y=5      列=宽=w=x=6    横轴x,纵轴y   最后变换完成,两组数据相同
a0=[[0,1,2,3,4,5],[10,11,12,13,14,15],[20,21,22,23,24,25],[30,31,32,33,34,35],[40,41,42,43,44,45]]  # 原始数据 5行 6列:h=5,w=6
a=[a0,a0,a0]  # 3通道 3*5*6
a_npy=np.array(a)          # npy格式a 3*5*6    作为image用       3*h*w
b_npy=np.array(a0)         # npy格式a0  5*6    作为mp用(密度图)   h*w
a_tensor=torch.Tensor(a)   # tensor格式a 3*5*6 作为image用       3*h*w
# ================转换方法===============================================================
a_pil = transforms.ToPILImage()(a_tensor)             # 转换为PIL格式 6×5  image mode=RGB                  w*h
a_pil_transforms_tensor=transforms.ToTensor()(a_pil)  # 将PIL格式(6*5)用transforms方法转换为tensor 3*5*6    3*h*w
a_transforms_tensor=transforms.ToTensor()(a_npy)      # 将npy格式(3*5*6)用transforms方法转换为tensor 6*3*5  w*3*h
# =================保存文件=============================================================
A_path='a.jpg'
B_path='a.npy'
scipy.misc.imsave(A_path, a_npy.transpose(1,2,0))  # a_npy(3*4*5)保存为jpg图像 宽度5,高度4
np.save(B_path, b_npy)
# =================读取文件===之前a,之后A==========================================================
A_img = Image.open(A_path).convert('RGB')  # 5*4  w*h
B_npy = read_npy(B_path)  # 3*4*5                 3*h*w
# ================ crop 2行4列 h=2,w=4  ================================================
    # 方法1 先切再转换为tensor----------------------------------------
x = 2  # random.randint(0, 6-4)  # w  先假定x=2,y=1
y = 1  # random.randint(0, 5-2)  # h  应该切出的值 12 13 14 15; 22 23 24 25
A_img_crop = Lcrop.LocationCrop((4, 2))(A_img, x, y)  #    w*h    4*2   'RGB'
B_npy_crop = Lcrop.LocationCropNpy((2, 4))(B_npy, y, x)  # 3*h*w  3*2*4  切出的值 12 13 14 15; 22 23 24 25
B_npy_crop2 = B_npy_crop.transpose(1, 2, 0) # h*w*3  2*4*3  切出的值 12 13 14 15; 22 23 24 25
        # transforms.ToTensor()-------------------------------------
A_img_crop_tensor = transforms.ToTensor()(A_img_crop)  # 3*2*4
B_npy_crop2_tensor = transforms.ToTensor()(B_npy_crop2)  # 3*2*4
    # 方法2 先转换为tensor 再切
# ================= 水平翻转 ==================================================================
# if random.random() < 0.5:
A_hflip = hflip_T(A_img_crop_tensor)
B_hflip = hflip_T(B_npy_crop2_tensor)

print('ok')



