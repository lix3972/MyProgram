
import torch
# import torch.nn as nn
# import numpy as np
import os
import scipy
from itertools import chain
from data_process import *
from DiscoModel import *
import matplotlib.pyplot as plt

# import torch.autograd as Variable
cuda=True
BATCH_SIZE = 20
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
imageSize = 64
update_interval=3
gan_curriculum=10000
starting_rate=0.01
default_rate=0.5
model_save_interval=10000
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

testPathA='dataset/test_im'
testPathB='dataset/test_npy'
test_A=get_filenames(testPathA)
test_B=get_npy_filenames(testPathB)
test_A = read_images(test_A, image_size=imageSize)
test_B = read_npy(test_B, image_size=imageSize)
test_A = torch.Tensor(test_A)
test_B = torch.Tensor(test_B)
test_A = test_A.cuda()
test_B = test_B.cuda()

filePathA = 'dataset/train_im'
filePathB = 'dataset/train_npy'
classA = get_filenames(filePathA)
classB = get_npy_filenames(filePathB)

classA = read_images(classA, image_size=imageSize)
classB = read_npy(classB, image_size=imageSize)
# print(classA.shape)
# exit()
n_batches= len(classA) // BATCH_SIZE
iters = 0

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
    labels_dis_real = (torch.ones( [dis_real.size()[0], 1] ))
    labels_dis_fake = (torch.zeros([dis_fake.size()[0], 1] ))
    labels_gen = (torch.ones([dis_fake.size()[0], 1]))

    if cuda:
        labels_dis_real = labels_dis_real.cuda()
        labels_dis_fake = labels_dis_fake.cuda()
        labels_gen = labels_gen.cuda()

    dis_loss = criterion( dis_real, labels_dis_real ) * 0.5 + criterion( dis_fake, labels_dis_fake ) * 0.5
    gen_loss = criterion( dis_fake, labels_gen )

    return dis_loss, gen_loss


for epoch in range(5000):

    shuffleA, shuffleB = shuffle_data(classA, classB)
    print('shuffiled classA and classB')

    for i in range(n_batches):
        n_star = i * BATCH_SIZE  #len(classA)/BATCH_SIZE = 400/40 = 10
        n_end = min(( i + 1) * BATCH_SIZE, len(classA))
        batchA = shuffleA[n_star:n_end]
        batchB = shuffleB[n_star:n_end]
        # loopN = min(BATCH_SIZE, n_end - n_star)

        # for i in range(loopN):

        A = torch.Tensor(batchA)  # artist_works_A()  # real painting from artist
        # A = A0.unsqueeze(0)
        A = A.cuda()

        B = torch.Tensor(batchB)  # artist_works_B()
        # B = B0.unsqueeze(0)
        B = B.cuda()


        AB = G_B(A)
        BA = G_A(B)

        ABA = G_A(AB)
        BAB = G_B(BA)




        # gen first
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

        G_A.zero_grad()
        G_B.zero_grad()
        D_A.zero_grad()
        D_B.zero_grad()
        opt_D.zero_grad()
        opt_G.zero_grad()
        if iters % update_interval == 0:
            dis_loss.backward()
            opt_D.step()
        else:
            gen_loss.backward()
            opt_G.step()

        if iters % 50 == 0:
            print("---------------------")

            print("GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean()))
            print("Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean()))
            print("RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean()))
            print("DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean()))

        if iters % 1000 == 0:
            AB = G_B(test_A)
            BA = G_A(test_B)
            ABA = G_A(AB)
            BAB = G_B(BA)

            n_testset = min(test_A.size()[0], test_B.size()[0])

            subdir_path = os.path.join(result_path, str(iters / 1000))

            if os.path.exists(subdir_path):
                pass
            else:
                os.makedirs(subdir_path)

            for im_idx in range(n_testset):
                A_val = test_A[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                B_val = test_B[im_idx].cpu().data.numpy().transpose(1, 2, 0)  # * 255.
                BA_val = BA[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                ABA_val = ABA[im_idx].cpu().data.numpy().transpose(1, 2, 0) * 255.
                AB_val = AB[im_idx].cpu().data.numpy().transpose(1, 2, 0)  #* 255.
                BAB_val = BAB[im_idx].cpu().data.numpy().transpose(1, 2, 0)  #* 255.

                # plt.imshow(B_val[:, :, 0], label='B')
                # plt.pause(1)
                filename_prefix = os.path.join(subdir_path, str(im_idx))
                scipy.misc.imsave(filename_prefix + '.A.jpg', A_val.astype(np.uint8)[:, :, ::-1])
                # scipy.misc.imsave(filename_prefix + '.B.jpg', B_val.astype(np.uint8)[:, :, 1])
                np.save(filename_prefix + '.B.npy', B_val.astype(np.float64)[:, :, 0])
                scipy.misc.imsave(filename_prefix + '.BA.jpg', BA_val.astype(np.uint8)[:, :, ::-1])
                # scipy.misc.imsave(filename_prefix + '.AB.jpg', AB_val.astype(np.uint8)[:, :, 1])
                np.save(filename_prefix + '.AB.npy', AB_val.astype(np.float64)[:, :, 0])
                scipy.misc.imsave(filename_prefix + '.ABA.jpg', ABA_val.astype(np.uint8)[:, :, ::-1])
                # scipy.misc.imsave(filename_prefix + '.BAB.jpg', BAB_val.astype(np.uint8)[:, :, 1])
                np.save(filename_prefix + '.BAB.npy', BAB_val.astype(np.float64)[:, :, 0])
        if iters % model_save_interval == 0:
            torch.save(G_A, os.path.join(model_path, 'model_gen_A-' + str(iters / model_save_interval)))
            torch.save(G_B, os.path.join(model_path, 'model_gen_B-' + str(iters / model_save_interval)))
            torch.save(D_A, os.path.join(model_path, 'model_dis_A-' + str(iters / model_save_interval)))
            torch.save(D_B, os.path.join(model_path, 'model_dis_B-' + str(iters / model_save_interval)))

        iters +=1

        # opt_G.zero_grad()
        # G_loss.backward()
        # opt_G.step()

    # if i == BATCH_SIZE - 1:  # plotting
    #     print('step ', step, ' finished.')
    #     print('    G_loss=', G_loss)
    #     print('    D_loss=', D_loss)
    #
    # if step==300:  # step%1000 == 999:
    #     G_A_path = 'pklFiles/G_A{}.pkl'.format(step)
    #     G_B_path = 'pklFiles/G_B{}.pkl'.format(step)
    #     D_A_path = 'pklFiles/D_A{}.pkl'.format(step)
    #     D_B_path = 'pklFiles/D_B{}.pkl'.format(step)
    #     torch.save(G_A.state_dict(), G_A_path)
    #     torch.save(G_B.state_dict(), G_B_path)
    #     torch.save(D_A.state_dict(), D_A_path)
    #     torch.save(D_B.state_dict(), D_B_path)

