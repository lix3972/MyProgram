###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

import torch
from PIL import Image
import os
import os.path
import numpy as np
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_npy_file(filename):
    return filename.endswith('.npy')

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def get_npy_filenames(dir):
    npys=[]
    assert os.path.isdir(dir), '%s is not a valid npy file directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_npy_file(fname):
                path = os.path.join(root,fname)
                npys.append(path)
    return npys

def read_npy(fileName):
    # read npy and transform to PILImage w*h
    dm = np.load(fileName) # ndarray 768*768
    dm = np.tile(dm, (3, 1, 1))  # nparray  3 w*h
    dm = dm.transpose(2, 1, 0)   # nparray  h*w 3
    dm = dm * 255. / 0.14721
    dm = transforms.ToPILImage()(dm.astype(np.uint8))  # PILImage w*h  mod='RGB'
    dm = dm.convert('RGB')
    return dm

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
