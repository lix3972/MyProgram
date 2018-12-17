
import torch
# import torch.nn as nn
# import numpy as np
import os
import scipy
from itertools import chain
from data_process import *
from DiscoModel import *
# import matplotlib.pyplot as plt
# import torch.autograd as Variable
cuda=True
# BATCH_SIZE = 1
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
imageSize = 720
update_interval=3
gan_curriculum=10000
starting_rate=0.01
default_rate=0.5
model_save_interval=1000
result_path='./results/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
model_path='./models/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
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

def as_np(data):
    return data.cpu().data.numpy()

def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion( l2, ( torch.ones( l2.size() ) ).cuda() )
        losses += loss

    return losses

def get_gan_loss(dis_real, dis_fake, criterion, cuda):
    labels_dis_real = (torch.ones( [dis_real.size()[0], 1,1,1] ))
    labels_dis_fake = (torch.zeros([dis_fake.size()[0], 1,1,1] ))
    labels_gen = (torch.ones([dis_fake.size()[0], 1,1,1]))

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.5 + criterion( dis_fake, labels_dis_fake ) * 0.5
    gen_loss = criterion( dis_fake, labels_gen )

    return dis_loss, gen_loss

# testPathA='dataset/test_im'
# testPathB='dataset/test_npy'

filePathA = 'dataset/train_im'
filePathB = 'dataset/train_npy'
iters=0

epochNum=200000
for epoch in range(epochNum):

    # shuffleA, shuffleB = shuffle_data(classA, classB)
    num=(epoch%400)+1
    fileName=os.path.join('dataset/train_im','IMG_{}.jpg'.format(num))
    imageA=read_image(fileName=fileName)  #3*720*720
    imageA=cropDataTo3_3(imageA)
    imageA=torch.Tensor(imageA)
    # A=torch.Tensor(A).unsqueeze(0)
    npyName=os.path.join('dataset/train_npy','IMG_{}.npy'.format(num))
    npyB = read_npy(npyName)
    npyB = cropDataTo3_3(npyB)
    npyB = torch.Tensor(npyB)
# ===训练==训练==训练===训练==训练===================================================================================
    for i in range(9):
        A = imageA[i].unsqueeze(0).cuda()
        B = npyB[i].unsqueeze(0).cuda()

        G_A.zero_grad()
        G_B.zero_grad()
        D_A.zero_grad()
        D_B.zero_grad()

        AB = G_B(A)
        BA = G_A(B)

        ABA = G_A(AB)
        BAB = G_B(BA)

        recon_loss_A = recon_criterion(ABA, A)  # mse
        recon_loss_B = recon_criterion(BAB, B)
        # temp0,temp1=D_B(AB)
        # L_GAB = -torch.log(D_B(AB))  # AB=G_B(A)
        A_dis_real, A_feats_real = D_A(A)
        A_dis_fake, A_feats_fake = D_A(BA)
        dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_criterion, cuda)
        fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion)

        # Real/Fake GAN Loss (B)
        B_dis_real, B_feats_real = D_B(B)
        B_dis_fake, B_feats_fake = D_B(AB)

        dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, gan_criterion, cuda)
        fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_criterion)
        if iters < gan_curriculum:
            rate = starting_rate
        else:
            rate = default_rate

        gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
        gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

        gen_loss = gen_loss_A_total + gen_loss_B_total
        dis_loss = dis_loss_A + dis_loss_B

        # opt_D.zero_grad()
        # opt_G.zero_grad()

        if iters % update_interval == 0:
            dis_loss.backward()
            opt_D.step()
        else:
            gen_loss.backward()
            opt_G.step()
        print(iters)
        iters += 1

    if epoch % 30 == 30 - 1:
        print("---------------------")

        print("GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean()))
        print("Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean()))
        print("RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean()))
        print("DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean()))

    if epoch == epochNum-1:
        torch.save(G_A.state_dict(), 'pklFiles/G_A{}.pkl'.format(epoch))
        torch.save(G_B.state_dict(), 'pklFiles/G_B{}.pkl'.format(epoch))
        torch.save(D_A.state_dict(), 'pklFiles/D_A{}.pkl'.format(epoch))
        torch.save(D_B.state_dict(), 'pklFiles/D_B{}.pkl'.format(epoch))
    elif epoch % model_save_interval == 0:
        torch.save(G_A.state_dict(), 'pklFiles/G_A{}.pkl'.format(epoch))
        torch.save(G_B.state_dict(), 'pklFiles/G_B{}.pkl'.format(epoch))
        torch.save(D_A.state_dict(), 'pklFiles/D_A{}.pkl'.format(epoch))
        torch.save(D_B.state_dict(), 'pklFiles/D_B{}.pkl'.format(epoch))

    # show the result
    if epoch % 1000 == 1000-1:
        A1=read_image('dataset/test_im/IMG_1.jpg')
        imageA = cropDataTo3_3(A1)
        imageA = torch.Tensor(imageA)

        B1 = read_npy('dataset/test_npy/IMG_1.npy')
        npyB = cropDataTo3_3(B1)
        npyB = torch.Tensor(npyB)
        AB1 = []
        BA1 = []
        ABA1 = []
        BAB1 = []
        for i in range(9):
            A = imageA[i].unsqueeze(0).cuda()
            B = npyB[i].unsqueeze(0).cuda()
            G_A.zero_grad()
            G_B.zero_grad()

            AB = G_B(A)
            BA = G_A(B)

            ABA = G_A(AB)
            BAB = G_B(BA)

            tmp = AB.squeeze(0).cpu().data.numpy()
            AB1.append(tmp)
            tmp = BA.squeeze(0).cpu().data.numpy()
            BA1.append(tmp)
            ABA1.append(ABA.squeeze(0).cpu().data.numpy())
            BAB1.append(BAB.squeeze(0).cpu().data.numpy())

        AB1 = combo3_3to1(AB1)
        BA1 = combo3_3to1(BA1)
        ABA1 = combo3_3to1(ABA1)
        BAB1 = combo3_3to1(BAB1)

        # A_val = A1.transpose(1, 2, 0) * 255.
        # B_val = B1.transpose(1, 2, 0)
        BA_val = BA1.transpose(1, 2, 0) * 255.
        AB_val = AB1.transpose(1, 2, 0)
        ABA_val = ABA1.transpose(1, 2, 0) * 255.
        BAB_val = BAB1.transpose(1, 2, 0)

        # scipy.misc.imsave('trainResults/' + str(int(epoch)) + '_A.jpg', A_val.astype(np.uint8)[:, :, :])  # [:, :, ::-1])
        # np.save('trainResults/' + str(int(epoch)) + '_B.npy', B_val.astype(np.float64)[:, :, 0])
        scipy.misc.imsave('trainResults/' + str(int(epoch)) + '_BA.jpg', BA_val.astype(np.uint8)[:, :, :])
        np.save('trainResults/' + str(int(epoch)) + '_AB.npy', AB_val.astype(np.float64)[:, :, 0])
        scipy.misc.imsave('trainResults/' + str(int(epoch)) + '_ABA.jpg', ABA_val.astype(np.uint8)[:, :, :])
        np.save('trainResults/' + str(int(epoch)) + '_BAB.npy', BAB_val.astype(np.float64)[:, :, 0])
