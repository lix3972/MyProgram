from __future__ import print_function
import torch
import torch.nn as nn


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':  # padding_type == 'reflect'
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:   # use_dropout = False
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


input_nc=6
model = [nn.ReflectionPad2d(3),    # left,right,top,bottom +3
         nn.Conv2d(input_nc, 64, kernel_size=7, padding=0,
                   bias=False),
         nn.InstanceNorm2d(64),
         nn.ReLU(True)]

n_downsampling = 2
for i in range(n_downsampling):
    mult = 2 ** i
    model += [nn.Conv2d(64 * mult, 64 * mult * 2, kernel_size=3,
                        stride=2, padding=1, bias=False),
              nn.InstanceNorm2d(64 * mult * 2),
              nn.ReLU(True)]

padding_type = 'reflect'
n_blocks = 6
for i in range(n_blocks):
    model += [ResnetBlock(64 * 4, padding_type=padding_type, norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=False)]

model += [nn.Conv2d(64 * 4, 64 * 4, 3, 2, 1, bias=False),  # 44 => 22
    nn.InstanceNorm2d(64 * 4),
    nn.LeakyReLU(0.2, inplace=True)]

model += [nn.Conv2d(64 * 4, 64 * 2, 3, 2, 1, bias=False),  # 22 => 11
    nn.InstanceNorm2d(64 * 2),
    nn.LeakyReLU(0.2, inplace=True)]

# model += [nn.Conv2d(64 * 2, 64, 3, 2, 1, bias=False),  # 22 => 11
#           nn.InstanceNorm2d(64),
#           nn.LeakyReLU(0.2, inplace=True)]

model += [nn.ReflectionPad2d(3),
          nn.Conv2d(64, 8, kernel_size=7, padding=0),
          nn.Tanh()]

localization = nn.Sequential(*model)  # 8*11*11

inp = torch.ones(1, 6, 176, 176)
out = localization(inp)

print('ok')

