import os
import pandas as pd
#import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
#from scipy.misc import imresize
#import scipy.io

def get_filenames(filePath):
    fileNames = map(lambda x: os.path.join(filePath, x), os.listdir( filePath ))
    fileNames = filter(lambda x: x.endswith('.jpg'), fileNames)
    fileNames = list(fileNames)
    file_idx = map(lambda x: int(x.split('IMG_')[1].split('.jpg')[0]), fileNames)
    file_idx = list(file_idx)
    file_df = pd.DataFrame({'idx': file_idx, 'path': fileNames}).sort_values(by='idx')
    fileNames = file_df['path'].values
    return fileNames

def get_npy_filenames(filePath):
    fileNames = map(lambda x: os.path.join(filePath, x), os.listdir( filePath ))
    fileNames = filter(lambda x: x.endswith('.npy'), fileNames)
    fileNames = list(fileNames)
    file_idx = map(lambda x: int(x.split('IMG_')[1].split('.npy')[0]), fileNames)
    file_idx = list(file_idx)
    file_df = pd.DataFrame({'idx': file_idx, 'path': fileNames}).sort_values(by='idx')
    fileNames = file_df['path'].values
    return fileNames

def read_images( fileNames, image_size=64):
    images = []
    with tf.Session() as sess:
        for fn in fileNames:

            image_raw = tf.gfile.FastGFile(fn, 'rb').read()
            image = tf.image.decode_jpeg(image_raw)

            if image is None:
                continue
            im = tf.image.resize_images(image, [image_size, image_size], method=0)
            image = im.eval()  # ndarray #ndarray 720*720*3
            image = image.astype(np.float32) / 255.
            image = image.transpose(2, 0, 1)
            images.append(image)
    if image_size > 1 :
        images = np.stack( images )
    return images

def read_npy(fileNames, image_size=64):
    images=[]
    with tf.Session() as sess:
        for fn in fileNames:
            # im_full_path = filepath + "/" + fn
            dm = np.load(fn) #ndarray 720*720
            dm = np.tile(dm, (3, 1, 1))
            dm = dm.transpose(1, 2, 0)
            im = tf.image.resize_images(dm, [64, 64])
            image = im.eval()

            if im is None:
                continue
            image = image.astype(np.float32) / 255.
            image = image.transpose(2,0,1)
            images.append( image )
    if image_size > 1:
        images = np.stack( images )
    return images

# a=get_filenames('dataset/train_im')
# a1=list(a)
# print(a1)
# img=read_images(a,720)
# print(img.shape)
# plt.imshow(img[0])
# plt.show()

# d= get_npy_filenames('dataset/train_npy')
# # d1=list(d)
# print(d[0])
# im=np.load(d[0])
# print(im.shape)
# plt.imshow(im)
# plt.show()