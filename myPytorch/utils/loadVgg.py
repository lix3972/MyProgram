import torch
import torch.nn as nn
import numpy as np
import scipy
import tensorflow as tf
# from vgg_feature import VGG2
import torchvision

# slim = tf.contrib.slim
# modelPath='./vgg2_model/vgg_2.ckpt'
# sess = tf.Session()
def loadVgg2():
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False)
    )
    model_dict = model.state_dict()
    vggNet=torchvision.models.vgg16(pretrained=True)
    pretrained_dict = vggNet.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return  model


