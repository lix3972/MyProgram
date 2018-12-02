import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from data_process import *
# import torch.autograd as Variable

BATCH_SIZE = 10
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
# N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
# ART_COMPONENTS = 50  # it could be total point G can draw in the canvas
# PAINT_POINTS = np.vstack([np.linspace(0, 2*3.14, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
# x = np.linspace(0, 2 * 3.14, ART_COMPONENTS)


# def artist_works_A():  # sin  1<a<2
#     a = np.random.uniform(1, 2, size=1)  # size=BATCH_SIZE) #[:, np.newaxis]
#     # w = np.random.uniform(0, 50, size=BATCH_SIZE)[:, np.newaxis]
#     # fy=np.random.uniform(0, 2*3.14, size=BATCH_SIZE)[:, np.newaxis]
#
#     y = a * np.sin(x)
#     y = torch.from_numpy(y).float()
#     return y
#
#
# def artist_works_B():  # cos 1<a<2
#     a = np.random.uniform(1, 2, size=1)  # size=BATCH_SIZE) #[:, np.newaxis]
#     # w = np.random.uniform(0, 50, size=BATCH_SIZE)[:, np.newaxis]
#     # fy=np.random.uniform(0, 2*3.14, size=BATCH_SIZE)[:, np.newaxis]
#     # x = np.linspace(0, 2 * 3.14, ART_COMPONENTS)
#     y = a * np.cos(x + 0)
#     y = torch.from_numpy(y).float()
#     return y

# torch.cuda.set_device(0)
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

G_A = G_A.cuda()
G_B = G_B.cuda()
D_A = D_A.cuda()
D_B = D_B.cuda()
G_params = chain(G_A.parameters(), G_B.parameters())
D_params = chain(D_A.parameters(), D_B.parameters())
opt_D = torch.optim.Adam(D_params, lr=LR_D)
opt_G = torch.optim.Adam(G_params, lr=LR_G)

recon_criterion = nn.MSELoss()
gan_criterion = nn.BCELoss()
feat_criterion = nn.HingeEmbeddingLoss()

plt.ion()  # something about continuous plotting

filePathA = 'dataset/train_im'
filePathB = 'dataset/train_npy'
classA=get_filenames(filePathA)
classB=get_npy_filenames(filePathB)

classA=read_images(classA,image_size=64)
classB=read_npy(classB)

for step in range(100):

    # classA = read_images(filePathA, image_size=64)
    # classB = read_npy(filePathB)
    for i in range(BATCH_SIZE):
        A0 = torch.Tensor(classA[i]) #artist_works_A()  # real painting from artist
        A = A0.unsqueeze(0)
        # A = Variable(A)
        A = A.cuda()
        B0 = torch.Tensor(classB[i]) #artist_works_B()
        B = B0.unsqueeze(0)
        # B = Variable(B)
        B = B.cuda()
        # G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
        # G_paintings = G(G_ideas)                    # fake painting from G (random ideas)

        # prob_artist0 = D(artist_paintings)          # D try to increase this prob
        # prob_artist1 = D(G_paintings)               # D try to reduce this prob
        AB = G_B(A)
        BA = G_A(B)

        ABA = G_A(AB)
        BAB = G_B(BA)

        # D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        # G_loss = torch.mean(torch.log(1. - prob_artist1))
        # gen first
        recon_loss_A = recon_criterion(ABA, A)  # mse
        L_GAB = -torch.log(D_B(G_B(A)))  # AB=G_B(A)
        # gen second
        recon_loss_B = recon_criterion(BAB, B)
        L_GBA = -torch.log(D_A(G_A(B)))  # BA=G_A(B)
        # gen total A B
        G_loss = (recon_loss_A + L_GAB + recon_loss_B + L_GBA).cuda()

        # dis
        L_DB = -torch.log(D_B(B)) - torch.log(1. - D_B(AB))  # AB=G_B(A)
        L_DA = -torch.log(D_A(A)) - torch.log(1. - D_A(BA))  # BA=G_A(B)
        D_loss = (L_DB + L_DA).cuda()

        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)  # reusing computational graph
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()



        if i == BATCH_SIZE - 1:  # plotting
            print('G_loss=',G_loss)
            print('D_loss=',D_loss)
#             plt.cla()
#             plt.subplot(231)
#             plt.cla()
#             plt.plot(x, A.data.numpy(), c='#4AD631', lw=3, label='sin', )
#             plt.draw()
#             plt.pause(0.01)
#
#             plt.subplot(232)
#             plt.cla()
#             plt.plot(x, AB.data.numpy(), c='#74BCFF', lw=3, label='AB', )
#             plt.draw()
#             plt.pause(0.01)
#
#             plt.subplot(233)
#             plt.cla()
#             plt.plot(x, ABA.data.numpy(), c='#74BCFF', lw=3, label='ABA', )
#             plt.draw()
#             plt.pause(0.01)
#
#             plt.subplot(234)
#             plt.cla()
#             plt.plot(x, B.data.numpy(), c='#74BCFF', lw=3, label='cos', )
#             plt.draw()
#             plt.pause(0.01)
#
#             plt.subplot(235)
#             plt.cla()
#             plt.plot(x, BA.data.numpy(), c='#74BCFF', lw=3, label='BA', )
#             plt.draw()
#             plt.pause(0.01)
#
#             plt.subplot(236)
#             plt.cla()
#             plt.plot(x, BAB.data.numpy(), c='#74BCFF', lw=3, label='BAB', )
#
#             # plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
#             # plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
#             # plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
#             # plt.ylim((-2, 2));#plt.legend(loc='upper right', fontsize=10);
#             plt.draw();
#             plt.pause(0.01)
#
# plt.ioff()
# plt.show()