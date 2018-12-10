import torch
import torch.nn as nn
import torch.nn.functional as F

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.conv1=nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.relu1=nn.LeakyReLU(0.2, inplace=True)
        self.conv2=nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.bn2=nn.BatchNorm2d(64 * 2)
        self.relu2=nn.LeakyReLU(0.2, inplace=True)
        self.conv3=nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.bn3=nn.BatchNorm2d(64 * 4)
        self.relu3=nn.LeakyReLU(0.2, inplace=True)
        self.conv4=nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.bn4=nn.BatchNorm2d(64 * 8)
        self.relu4=nn.LeakyReLU(0.2, inplace=True)

        self.convT5=nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
        self.bn5=nn.BatchNorm2d(64 * 4)
        self.relu5=nn.ReLU(True)
        self.convT6=nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False)
        self.bn6=nn.BatchNorm2d(64 * 2)
        self.relu6=nn.ReLU(True)
        self.convT7=nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False)
        self.bn7=nn.BatchNorm2d(64)
        self.relu7=nn.ReLU(True)
        self.convT8=nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        #self.out=nn.Sigmoid()  # making a painting from these random ideas

    def forward(self,input):
        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)

        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu3(bn3)

        conv4 = self.conv4(relu3)
        bn4 = self.bn4(conv4)
        relu4 = self.relu4(bn4)

        convT5=self.convT5(relu4)
        bn5 = self.bn5(convT5)
        relu5 = self.relu5(bn5)

        convT6 =self.convT6(relu5)
        bn6 = self.bn6(convT6)
        relu6 = self.relu6(bn6)
        convT7= self.convT7(relu6)
        bn7 = self.bn7(convT7)
        relu7 = self.relu7(bn7)
        convT8 = self.convT8(relu7)

        Dout = torch.sigmoid(convT8)
        return Dout


class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64 * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64 * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        # self.conv5 = nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
        self.conv5 = nn.Conv2d(64 * 8, 1, 16, 1, 0, bias=False)
    def forward(self, input):
        conv1 = self.conv1( input )
        relu1 = self.relu1( conv1 )

        conv2 = self.conv2( relu1 )
        bn2 = self.bn2( conv2 )
        relu2 = self.relu2( bn2 )

        conv3 = self.conv3( relu2 )
        bn3 = self.bn3( conv3 )
        relu3 = self.relu3( bn3 )

        conv4 = self.conv4( relu3 )
        bn4 = self.bn4( conv4 )
        relu4 = self.relu4( bn4 )

        conv5 = self.conv5( relu4 )
        Dout = torch.sigmoid( conv5 )
        return Dout

