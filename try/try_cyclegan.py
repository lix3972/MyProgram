import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

input_nc = 6
output_nc = 3
ngf=64
norm_layer=nn.InstanceNorm2d
use_dropout=False
n_blocks=9
padding_type='reflect'
l1_1=nn.Sequential(nn.ReflectionPad2d(3))  # left,right,top,bottom +3
l1_2=nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=False),
                 norm_layer(ngf),
                 nn.ReLU(True))
l2=nn.Sequential(nn.Conv2d(64,64*2,3,2,1),
                 nn.InstanceNorm2d(64*2),
                 nn.ReLU(True))
l3=nn.Sequential(nn.Conv2d(64*2,64*4,3,2,1),
                 nn.InstanceNorm2d(64*4),
                 nn.ReLU(True))
inp = torch.ones(1, 6, 176, 176)
out1_1 = l1_1(inp)
out1_2 = l1_2(out1_1)
out2 = l2(out1_2)
out3 = l3(out2)

model_ft = models.resnet18()
print('ok')
