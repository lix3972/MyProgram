import os.path
# from data.base_dataset import BaseDataset
from data.image_folder import make_dataset, get_npy_filenames, read_npy
from PIL import Image
import torch.utils.data as data
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import data.LocationCrop as Lcrop
from utils.dataflip import *  # hflip_T

class CrowdDataset(data.Dataset):
    def __init__(self):
        super(CrowdDataset, self).__init__()
        # self.initialize()

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # datasets/crowd_2/+trainA or testA
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = get_npy_filenames(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        if self.A_size != self.B_size:
            print(opt.phase+'A.size muse equals to '+opt.phase+'B')

    def __getitem__(self, index):

        A_path = self.A_paths[index % min(self.A_size,self.B_size)]
        B_path = self.B_paths[index % min(self.A_size,self.B_size)]
        A_img = Image.open(A_path).convert('RGB')
        B_npy = read_npy(B_path, self.opt.npyscale)
        # =========crop256=================
        x = random.randint(1, 768 - 256)
        y = random.randint(1, 768 - 256)
        A_img = Lcrop.LocationCrop((256, 256))(A_img, x, y)  # w*h 'RGB'
        B_npy = Lcrop.LocationCropNpy((256, 256))(B_npy, x, y)  # 3*h*w
        B_npy = B_npy.transpose(1, 2, 0)  # h*w*3
        # if self.opt.phase and np.random.random() > 0.5:
        #     A_img = np.fliplr(A_img)
        #     B_npy = np.fliplr(B_npy)
        A=transforms.ToTensor()(A_img)  # 3*h*w
        A = A.type(torch.FloatTensor)
        B=transforms.ToTensor()(B_npy)  # 3*h*w
        B = B.type(torch.FloatTensor)
        if self.opt.phase == 'train' and random.random() < 0.5:
            A = hflip_T(A)
            B = hflip_T(B)
        # if input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #     A = tmp.unsqueeze(0)
        #
        # if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'CrowdDataset'
