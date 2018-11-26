import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE=64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 20             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 50     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(0, 2*3.14, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artist_works():     # painting from the famous artist (real target)
    #a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    #w = np.random.uniform(0, 50, size=BATCH_SIZE)[:, np.newaxis]
    #fy=np.random.uniform(0, 2*3.14, size=BATCH_SIZE)[:, np.newaxis]
    paintings = 1 * np.sin(PAINT_POINTS+0)
    paintings = torch.from_numpy(paintings).float()
    return paintings

class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        self.fc1 = nn.Linear(N_IDEAS,128)
        self.fc2 = nn.Linear(128,ART_COMPONENTS)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        self.fc1 = nn.Linear(ART_COMPONENTS, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, y):
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return y

G = Gen()
D = Disc()
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()           # real painting from artist
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
    G_paintings = G(G_ideas)                    # fake painting from G (random ideas)

    prob_artist0 = D(artist_paintings)          # D try to increase this prob
    prob_artist1 = D(G_paintings)               # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], artist_paintings.data.numpy()[0], c='#74BCFF', lw=3, label='upper bound')
       # plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((-2, 2));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

plt.ioff()
plt.show()
