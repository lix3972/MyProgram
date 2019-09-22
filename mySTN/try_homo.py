import torch
import torch.nn as nn
from utils.homo_transform import Homo_grid_generator
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.image_show import plt_tensor
from PIL import Image
import torchvision.transforms as transforms
from math import cos, sin, pi

if __name__ == '__main__':

    # theta = [[0.7,-0.7,0],[0.7,0.7,0],[0.2,0.2,1]]
    # theta = [[], [], []]
    # theta = [[1.66063, -0.41903, 0], [0.44614, 0.79480, 0], [-17.14486, -0.00025, 0.99981]]
    # dthetas = [[[0.0344, -0.0156, 0.1338], [0.1739,  0.0249, -0.0565], [-0.2489,  0.1464, 0.1132]],
    #            [[0.1974, -0.0323, 0.1579], [0.1789, -0.1735, 0.2193],  [-0.3078, 0.1930, 0.1544]],
    #            [[0.2326, -0.0436, 0.1718], [0.1890, -0.2147, 0.2664], [-0.3357, 0.2026, 0.1420]],
    #            [[0.2544, -0.0240, 0.1375], [0.1673, -0.2917, 0.3336], [-0.3440, 0.2210, 0.1582]],
    #            ]
    # pi = 3.14159265
    # 旋转 role
    rz = pi / 6
    ry = pi / 6
    rx = pi / 6
    h_rz = torch.tensor([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
    h_ry = torch.tensor([[cos(ry), 0, -sin(ry)], [0, 1, 0], [sin(ry), 0, cos(ry)]])
    h_rx = torch.tensor([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])

    # 缩放 scale
    sx = 2
    sy = 1.5
    h_s = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    # 复合矩阵
    # theta = h_ry.mm(h_rz).mm(h_rx)
    # theta = h_rx.mm(h_ry).mm(h_rz)
    theta = h_s.mm(h_rx).mm(h_ry).mm(h_rz)
    theta2 = h_s.mm(h_ry).mm(h_rz).mm(h_rx)
    H1 = torch.Tensor(theta).unsqueeze(0)/theta[2][2]
    H_yzx = torch.Tensor(theta2).unsqueeze(0) / theta2[2][2]
    print(H1)
    print(H_yzx)

    # 读取图片
    A_path = '/home/lix/mySTN10/datasets/datasets/My_hatted/trainA/001561.jpg'
    A_img = Image.open(A_path).convert('RGB')

    # 图片转成tensor
    transform_img = transforms.Compose([
        # transforms.CenterCrop(176),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    A = transform_img(A_img)
    A = A.unsqueeze(0)

    # 单应性变换
    grid = Homo_grid_generator(H1,  A.size(), device=None)
    y1 = F.grid_sample(A, grid)

    # 画图
    title = 'unknown'
    dety1 = torch.det(H1.squeeze(0))
    if dety1 == 0:
        y2 = y1
        title = 'False'
    else:
        H2 = torch.inverse(H1)
        grid2 = Homo_grid_generator(H2, A.size(), device=None)
        y2 = F.grid_sample(y1, grid2)
        title = 'OK'
    plt.subplot(131)
    plt_tensor(A, convert_image=True)
    plt.subplot(132)
    plt_tensor(y1, convert_image=True)
    plt.subplot(133)
    plt_tensor(y2, convert_image=True, title=title)
    plt.show()

    print('break')
print('Finished.')
