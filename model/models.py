import torch
import pretrainedmodels
import torch.nn as nn
from torch.nn import init
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np
import math


def get_efficientnet(model_name='efficientnet-b0', num_classes=2, pretrained=True):
    if pretrained:
        net = EfficientNet.from_pretrained(model_name)
    else:
        net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


if __name__ == '__main__':
    model, image_size = get_efficientnet(model_name='efficientnet-b0', num_classes=2, pretrained=True), 224

    model = model.to(torch.device('cpu'))
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s, device='cpu'))

    print(model._modules.items())


    pass

