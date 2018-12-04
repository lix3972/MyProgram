import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from data_process import *
import tensorflow as tf

G_A = nn.Sequential(  # Generator
    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 2),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 8),
    nn.LeakyReLU(0.2, inplace=True),

    nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 4),
    nn.ReLU(True),
    nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 2),
    nn.ReLU(True),
    nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(    64,      3, 4, 2, 1, bias=False),
    nn.Sigmoid()  # making a painting from these random ideas
)
G_B = nn.Sequential(  # Generator
    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 2),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 8),
    nn.LeakyReLU(0.2, inplace=True),

    nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 4),
    nn.ReLU(True),
    nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 2),
    nn.ReLU(True),
    nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
    nn.Sigmoid() # making a painting from these random ideas
)

D_A = nn.Sequential(  # Discriminator
    nn.Conv2d(3,64,4,2,1,bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 2),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 8),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
    nn.Sigmoid()
)
D_B = nn.Sequential(  # Discriminator
    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 2),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 4),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
    nn.BatchNorm2d(64 * 8),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
    nn.Sigmoid(),  # tell the probability that the art work is made by artist
)

# G_A = G_A.cuda()
# G_B = G_B.cuda()
# D_A = D_A.cuda()
# D_B = D_B.cuda()

G_A_path='pklFiles/G_A.pkl'
G_B_path='pklFiles/G_B.pkl'
D_A_path='pklFiles/D_A.pkl'
D_B_path='pklFiles/D_B.pkl'

# G_A = G_A()
G_A.load_state_dict(torch.load(G_A_path))
G_A.eval()

G_B.load_state_dict(torch.load(G_B_path))
G_B.eval()

# D_A.load_state_dict(torch.load(D_A_path))
# D_A.eval()
#
# D_B.load_state_dict(torch.load(D_B_path))
# D_B.eval()

filePathA = 'dataset/test_im'
filePathB = 'dataset/test_npy'
classA=get_filenames(filePathA)
classB=get_npy_filenames(filePathB)
classA=read_images(classA,image_size=720)
classB=read_npy(classB,image_size=720)

# plt.ion()   # something about continuous plotting

for i in range(10):
    A0 = torch.Tensor(classA[i])  # artist_works_A()  # real painting from artist
    A = A0.unsqueeze(0)
    # A = A.cuda()

    B0 = torch.Tensor(classB[i])  # artist_works_B()
    B = B0.unsqueeze(0)
    # B = B.cuda()

    AB = G_B(A)
    BA = G_A(B)
    ABA = G_A(AB)
    BAB = G_B(BA)

    plt.subplot(231)
    plt.cla()
    A = A.squeeze(0)
    # A=A.numpy()
    A = A.transpose(1,0)
    A = A.transpose(1,2)
    plt.imshow(A,label='A')

    plt.subplot(232)
    plt.cla()
    AB=AB.squeeze(0)
    AB = AB.transpose(1, 0)
    AB = AB.transpose(1, 2)
    AB = AB.data.cpu().numpy()
    plt.imshow(AB[:,:,0])

    plt.subplot(233)
    plt.cla()
    ABA = ABA.squeeze(0)
    ABA = ABA.transpose(1, 0)
    ABA = ABA.transpose(1, 2)
    ABA = ABA.data.cpu().numpy()
    plt.imshow(ABA)

    plt.subplot(234)
    plt.cla()
    B = B.squeeze(0)
    B = B.transpose(1, 0)
    B = B.transpose(1, 2)
    plt.imshow(B[:,:,0], label='B' )

    plt.subplot(235)
    plt.cla()
    BA = BA.squeeze(0)
    BA = BA.transpose(1, 0)
    BA = BA.transpose(1, 2)
    BA = BA.data.cpu().numpy()
    plt.imshow(BA , label='BA' )

    plt.subplot(236)
    plt.cla()
    BAB = BAB.squeeze(0)
    BAB = BAB.transpose(1, 0)
    BAB = BAB.transpose(1, 2)
    BAB = BAB.data.cpu().numpy()
    plt.imshow( BAB[:,:,0], label='BAB', )

    plt.pause(0.5)

