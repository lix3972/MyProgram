import os
import pandas as pd
#import cv2
import numpy as np
import tensorflow as tf
import torch
from skimage import io,data
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

def read_images( fileNames, image_size=720):
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

def read_image( fileName ):

    # with tf.Session() as sess:
    #     image_raw = tf.gfile.FastGFile(fileName, 'rb').read()
    #     image = tf.image.decode_jpeg(image_raw)
    #     # im = tf.image.resize_images(image, [image_size, image_size], method=0)
    #     image = image.eval()  # ndarray #ndarray 720*720*3
    #     image = image.astype(np.float32) / 255.
    #     image = image.transpose(2, 0, 1)
    image=io.imread(fileName)
    image=image.astype(np.float32) / 255.
    image=image.transpose(2,0,1)
    return image

def read_npys(fileNames, image_size=720):
    dms=[]
    # with tf.Session() as sess:
    for fn in fileNames:
        # im_full_path = filepath + "/" + fn
        dm = np.load(fn) #ndarray 720*720
        dm = np.tile(dm, (3, 1, 1))
        dm = dm.transpose(1, 2, 0)
        dm = tf.image.resize_images(dm, [image_size, image_size])
        dm = dm.eval()
        dm = dm.astype(np.float32)   # 255.
        dm = dm.transpose(2,0,1)
        dms.append( dm )
    if image_size > 1:
        dms = np.stack( dms )
    return dms

def read_npy(fileName):
    dm = np.load(fileName) #ndarray 720*720
    dm = np.tile(dm, (3, 1, 1))
    dm = dm.astype(np.float32)
    return dm

def shuffle_data(da, db):
    a_idx = np.arange(len(da))
    np.random.shuffle( a_idx )

    # b_idx = range(len(db))
    # np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[ np.array(a_idx) ]
    shuffled_db = np.array(db)[ np.array(a_idx) ]

    return shuffled_da, shuffled_db
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

def cropDataTo3_3(im,image_size=720):
    w=int(image_size / 3)
    h = int(image_size / 3)

    im1 = im[:, 0:w, 0:h]
    im2 = im[:, w:2*w, 0:h]
    im3 = im[:, 2*w:3*w, 0:h]
    im4 = im[:, 0:w, h:2*h]
    im5 = im[:, w:2*w, h:2*h]
    im6 = im[:, 2*w:3*w, h:2*h]
    im7 = im[:, 0:w, 2*h:3*h]
    im8 = im[:, w:2*w, 2*h:3*h]
    im9 = im[:, 2*w:3*w, 2*h:3*h]
    # fullim = torch.cat([im1, im2, im3, im4,im5,im6,im7,im8,im9],0)
    fullim = np.stack([im1, im2, im3, im4, im5, im6, im7, im8, im9], 0)
    return fullim

def combo3_3to1(crop,crop_size=240):
    # crop_val=crop.squeeze(0).cpu().data.numpy()
    crop_val = crop
    w=h=crop_size
    im=np.zeros([3,w*3,h*3])
    im[:, :w, :h]=crop_val[0]
    im[:, w:2 * w, :h]=crop_val[1]
    im[:, 2 * w:3 * w, :h]=crop_val[2]
    im[:, :w, h:2 * h]=crop_val[3]
    im[:, w:2 * w, h:2 * h]=crop_val[4]
    im[:, 2 * w:3 * w, h:2 * h]=crop_val[5]
    im[:, :w, 2 * h:3 * h]=crop_val[6]
    im[:, w:2 * w, 2 * h:3 * h]=crop_val[7]
    im[:, 2 * w:3 * w, 2 * h:3 * h]=crop_val[8]
    return im