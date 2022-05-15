import torch
import math
from torch import nn
import numpy as np
from .Lip_Layer import Dist_Conv2D, Dist_Conv2D_1, Uni_Conv2D
from .Basic_Layers import Static_Layernorm

class Dis2Dis3(nn.Module):

    def __init__(self, in_channels, out_channels, p=torch.inf, conn_num=3):
        super().__init__()
        och_1 = out_channels // 2
        och_2 = out_channels - och_1
        self.Dis2 = Dist_Conv2D(in_channels, och_1, kernel_size=(2, 2), p=p, conn_num=conn_num)
        self.Dis3 = Dist_Conv2D(in_channels, och_2, kernel_size=(3, 3), p=p, conn_num=conn_num)

    def forward(self, x):
        Dis2 = self.Dis2(x)
        Dis3 = self.Dis3(x)
        return torch.concat([Dis2, Dis3], dim=1)

class Dis1Dis3(nn.Module):

    def __init__(self, in_channels, out_channels, p=torch.inf, conn_num=3):
        super().__init__()
        och_1 = out_channels // 2
        och_2 = out_channels - och_1
        self.Dis1 = Dist_Conv2D_1(in_channels, och_1, conn_num=conn_num)
        self.Dis3 = Dist_Conv2D(in_channels, och_2, kernel_size=(3, 3), p=p, conn_num=conn_num)

    def forward(self, x):
        Dis1 = self.Dis1(x)
        Dis3 = self.Dis3(x)
        return torch.concat([Dis1, Dis3], dim=1)


class DisUni(nn.Module):

    def __init__(self, in_channels, out_channels, conn_dis=3, conn_uni=6, ker_dis=(3, 3), ker_uni=(3, 3), p=torch.inf):
        super().__init__()
        och_1 = out_channels * 2 // 3
        och_2 = out_channels - och_1
        self.Dis = Dist_Conv2D(in_channels, och_1, kernel_size=ker_dis, p=p, conn_num=conn_dis)
        self.Uni = Uni_Conv2D(in_channels, och_2, kernel_size=ker_uni, conn_num=conn_uni)

    def forward(self, x):
        Dis = self.Dis(x)
        Uni = self.Uni(x)
        return torch.concat([Dis, Uni], dim=1)