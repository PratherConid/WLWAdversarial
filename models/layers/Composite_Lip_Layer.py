import torch
import math
from torch import nn
import numpy as np
from .Lip_Layer import Dist_Conv2D, Uni_Conv2D, Minimax_Conv2D
from .Basic_Layers import Static_Layernorm

class Dis2Dis3(nn.Module):

    def __init__(self, in_channels, out_channels, p=torch.inf, conn_num=3):
        super().__init__()
        och_1 = out_channels // 2
        och_2 = out_channels - och_1
        self.Dis2 = Dist_Conv2D(in_channels, och_1, kernel_size=(2, 2), p=p, padding=1, conn_num=conn_num)
        self.Dis3 = Dist_Conv2D(in_channels, och_2, kernel_size=(3, 3), p=p, padding=2, conn_num=conn_num)

    def forward(self, x):
        Dis2 = self.Dis2(x)
        Dis3 = self.Dis3(x)
        return torch.concat([Dis2, Dis3], dim=1)