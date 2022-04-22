import torch
from models.layers.Lip_Layer import Dist_Conv2D
from models.CIFAR_10_DistConv import CIFAR_10_DistConv

inp = torch.rand(15, 3, 32, 32)
net = CIFAR_10_DistConv()
ourp = net(inp)