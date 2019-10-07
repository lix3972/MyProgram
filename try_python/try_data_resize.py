from data.image_folder import read_npy
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch


def npy_resize(npy, widths, heights):
    a, _, c = npy.shape
    if a == 3 and c != 3:
        npy = npy.transpose(1, 2, 0)
    max_n = npy.max()
    scale_n = 255 / max_n
    npy_255 = npy * scale_n
    img_255 = Image.fromarray(npy_255.astype('uint8'))
    img_re = img_255.resize((widths, heights))
    npy_re = np.array(img_re)
    npy_re = npy_re / scale_n
    if a == 3 and c != 3:
        npy_re = npy_re.transpose(2, 0, 1)
    return npy_re


def tensor_resize(tens, widths, heights):
    npy = tens.data.numpy()
    npy_re = npy_resize(npy, widths, heights)
    return torch.tensor(npy_re, dtype=torch.float32)


if __name__ == '__main__':
    for i in range(10):
        npy_path = '/home/lix/myCount3_ucf50/datasets/ucf50_03/trainB/{}.npy'.format(i+1)
        img0 = read_npy(npy_path)
        print(img0.shape)
        img = torch.tensor(img0)
        result_npy = npy_resize(img0, 384, 384)
        result_tens = tensor_resize(img, 384, 384)
        print(result_npy.shape)
        a = img0.sum()
        b = result_npy.sum() * 4
        c = result_tens.sum() * 4
        print('img.sum = {}, result_npy.sum * 4 = {}, result_tens = {}'.format(a, b, c))
        plt.subplot(131)
        plt.imshow(img0.mean(0), cmap=plt.get_cmap('jet'))
        plt.subplot(132)
        plt.imshow(result_npy.mean(0), cmap=plt.get_cmap('jet'))
        plt.subplot(133)
        result_tens2 = result_tens.data.numpy()
        plt.imshow(result_tens2.mean(0), cmap=plt.get_cmap('jet'))

        plt.show()

