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

# imageSize = 720

img_path='./results/img/'
if not os.path.exists(img_path):
    os.makedirs(img_path)
npy_path='./results/npy/'
if not os.path.exists(npy_path):
    os.makedirs(npy_path)

G_A_path='models/G_A159999.pkl'
G_B_path='models/G_B159999.pkl'


# G_A = Gen_A()
# G_B = Gen_B()

#
# G_A.load_state_dict(torch.load(G_A_path))
# G_A.eval()
#
# G_B.load_state_dict(torch.load(G_B_path))
# G_B.eval()
G_A=torch.load(G_A_path)
# G_A.eval()
G_A.train()

G_B=torch.load(G_B_path)
# G_B.eval()
G_B.train()

G_A = G_A.cuda()
G_B = G_B.cuda()

filePathA = 'dataset/test_im'
filePathB = 'dataset/test_npy'

mseB=0

mseBAB=0

for j in range(316):
    fileName = os.path.join(filePathA, 'IMG_{}.jpg'.format(j+1))
    A1 = read_image(fileName=fileName)  # 3*720*720
    imageA = cropDataTo3_3(A1)
    imageA = torch.Tensor(imageA)
    # A=torch.Tensor(A).unsqueeze(0)
    npyName = os.path.join(filePathB, 'IMG_{}.npy'.format(j+1))
    B1 = read_npy(npyName)
    npyB = cropDataTo3_3(B1)
    npyB = torch.Tensor(npyB)
    AB1=[]
    BA1=[]
    ABA1=[]
    BAB1=[]
    for i in range(9):
        A = imageA[i].unsqueeze(0).cuda()
        B = npyB[i].unsqueeze(0).cuda()

        AB = G_B(A)
        BA = G_A(B)

        ABA = G_A(AB)
        BAB = G_B(BA)

        tmp=AB.squeeze(0).cpu().data.numpy()
        AB1.append(tmp)
        tmp=BA.squeeze(0).cpu().data.numpy()
        BA1.append(tmp)
        ABA1.append(ABA.squeeze(0).cpu().data.numpy())
        BAB1.append(BAB.squeeze(0).cpu().data.numpy())

    AB1 = combo3_3to1(AB1)
    BA1 = combo3_3to1(BA1)
    ABA1 = combo3_3to1(ABA1)
    BAB1 = combo3_3to1(BAB1)

    A_val = A1.transpose(1, 2, 0) * 255.
    B_val = np.mean(B1.transpose(1, 2, 0),2)
    # B_val = B1.transpose(1, 2, 0)
    BA_val = BA1.transpose(1, 2, 0) * 255.
    AB_val = np.mean(AB1.transpose(1, 2, 0),2)
    # AB_val = AB1.transpose(1, 2, 0)
    ABA_val = ABA1.transpose(1, 2, 0) * 255.
    BAB_val = np.mean(BAB1.transpose(1, 2, 0),2)
    # BAB_val = BAB1.transpose(1, 2, 0)

    scipy.misc.imsave(img_path + str(int(j))+ '_A.jpg', A_val.astype(np.uint8)[:, :, :])  # [:, :, ::-1])
    np.save(npy_path +str(int(j))+ '_B.npy', B_val.astype(np.float64)[:, :])
    scipy.misc.imsave(img_path + str(int(j))+ '_BA.jpg', BA_val.astype(np.uint8)[:, :, :])
    np.save(npy_path +str(int(j))+ '_AB.npy', AB_val.astype(np.float64)[:, :])
    scipy.misc.imsave(img_path + str(int(j))+ '_ABA.jpg', ABA_val.astype(np.uint8)[:, :, :])
    np.save(npy_path +str(int(j))+ '_BAB.npy', BAB_val.astype(np.float64)[:, :])

    print('IMG_',j,' : ')
    print('     B  = ',sum(sum(B_val)))
    print('     AB = ',sum(sum(AB_val)))
    print('     BAB= ',sum(sum(BAB_val)))
    mseB += (sum(sum(AB_val))-sum(sum(B_val)))*(sum(sum(AB_val))-sum(sum(B_val)))

    mseBAB += (sum(sum(BAB_val))-sum(sum(B_val)))*(sum(sum(BAB_val))-sum(sum(B_val)))
    # plt.subplot(231)
    # plt.cla()
    # A = A.squeeze(0)
    # # A=A.numpy()
    # A = A.transpose(1,0)
    # A = A.transpose(1,2)
    # plt.imshow(A,label='A')
    #
    # plt.subplot(232)
    # plt.cla()
    # AB = AB.squeeze(0)
    # AB = AB.transpose(1, 0)
    # AB = AB.transpose(1, 2)
    # AB = AB.data.cpu().numpy()
    # plt.imshow(AB[:,:,0])
    #
    # plt.subplot(233)
    # plt.cla()
    # ABA = ABA.squeeze(0)
    # ABA = ABA.transpose(1, 0)
    # ABA = ABA.transpose(1, 2)
    # ABA = ABA.data.cpu().numpy()
    # plt.imshow(ABA)
    #
    # plt.subplot(234)
    # plt.cla()
    # B = B.squeeze(0)
    # B = B.transpose(1, 0)
    # B = B.transpose(1, 2)
    # plt.imshow(B[:,:,0], label='B' )
    #
    # plt.subplot(235)
    # plt.cla()
    # BA = BA.squeeze(0)
    # BA = BA.transpose(1, 0)
    # BA = BA.transpose(1, 2)
    # BA = BA.data.cpu().numpy()
    # plt.imshow(BA , label='BA' )
    #
    # plt.subplot(236)
    # plt.cla()
    # BAB = BAB.squeeze(0)
    # BAB = BAB.transpose(1, 0)
    # BAB = BAB.transpose(1, 2)
    # BAB = BAB.data.cpu().numpy()
    # plt.imshow( BAB[:,:,0], label='BAB', )
    #
    # plt.pause(1)
# mseB=mseB.cpu().data.numpy()
# mseBAB=mseBAB.cpu().data.numpy()
mseB = np.sqrt(mseB/316)
mseBAB = np.sqrt(mseBAB/316)
print('mseB=',mseB)
print('mseBAB=',mseBAB)