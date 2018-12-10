
# import torch
# import torch.nn as nn
# import numpy as np

from itertools import chain
from data_process import *
from DiscoModel import *

# import torch.autograd as Variable

BATCH_SIZE = 10
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
imageSize = 64
# torch.cuda.set_device(0)
G_A = Gen()
G_B = Gen()

D_A = Disc()
D_B = Disc()

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

filePathA = 'dataset/train_im'
filePathB = 'dataset/train_npy'
classA = get_filenames(filePathA)
classB = get_npy_filenames(filePathB)

classA = read_images(classA, image_size=imageSize)
classB = read_npy(classB, image_size=imageSize)

for step in range(100):

    for i in range(BATCH_SIZE):
        A0 = torch.Tensor(classA[i])  # artist_works_A()  # real painting from artist
        A = A0.unsqueeze(0)
        A = A.cuda()

        B0 = torch.Tensor(classB[i])  # artist_works_B()
        B = B0.unsqueeze(0)
        B = B.cuda()

        AB = G_B(A)
        BA = G_A(B)

        ABA = G_A(AB)
        BAB = G_B(BA)

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
            print('G_loss=', G_loss)
            print('D_loss=', D_loss)
G_A_path = 'pklFiles/G_A.pkl'
G_B_path = 'pklFiles/G_B.pkl'
D_A_path = 'pklFiles/D_A.pkl'
D_B_path = 'pklFiles/D_B.pkl'
torch.save(G_A.state_dict(), G_A_path)
torch.save(G_B.state_dict(), G_B_path)
torch.save(D_A.state_dict(), D_A_path)
torch.save(D_B.state_dict(), D_B_path)

