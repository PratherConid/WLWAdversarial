import torch
from torch import nn
from .layers.Lip_Layer import Dist_Conv2D, Uni_Linear
from .layers.Basic_Layers import Static_Layernorm, MovingAverage

class CIFAR_10_DistConv(nn.Module):
    def __init__(self, conn_num, p=torch.inf):
        super(CIFAR_10_DistConv, self).__init__()
        self.DC1_1 = Dist_Conv2D(3, 256, padding=2, p=p, conn_num=conn_num)
        self.MA1_1 = MovingAverage(ax=(0, -1, -2))
        self.DC1_2 = Dist_Conv2D(256, 256, padding=2, p=p, conn_num=conn_num)
        self.SL1 = Static_Layernorm(ax=(0, -1, -2))
        self.mp1 = nn.MaxPool2d(2, 2)

        self.DC2_1 = Dist_Conv2D(256, 1024, padding=2, p=p, conn_num=conn_num)
        self.MA2_1 = MovingAverage(ax=(0, -1, -2))
        self.DC2_2 = Dist_Conv2D(1024, 1024, padding=2, p=p, conn_num=conn_num)
        self.SL2 = Static_Layernorm(ax=(0, -1, -2))
        self.mp2 = nn.MaxPool2d(2, 2)

        self.DC3_1 = Dist_Conv2D(1280, 2048, padding=2, p=p, conn_num=conn_num)
        self.MA3_1 = MovingAverage()
        self.DC3_2 = Dist_Conv2D(2048, 2048, padding=2, p=p, conn_num=conn_num)
        self.SL3 = Static_Layernorm()
        self.mp3 = nn.MaxPool2d(2, 2)

        self.DC4_1 = Dist_Conv2D(3328, 2048, padding=2, p=p, conn_num=conn_num)
        self.MA4_1 = MovingAverage()
        self.DC4_2 = Dist_Conv2D(2048, 1024, padding=2, p=p, conn_num=2 * conn_num)
        self.SL4 = Static_Layernorm()
        self.mp4 = nn.MaxPool2d(2, 2)

        self.head = nn.Sequential(
            nn.Flatten(), Uni_Linear(4096, 80), Static_Layernorm(),
            nn.ReLU(), Uni_Linear(80, 10),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B_1 = self.DC1_2(self.MA1_1(self.DC1_1(x)))
        BF_1 = self.mp1(self.SL1(B_1))

        B_2 = self.DC2_2(self.MA2_1(self.DC2_1(BF_1)))
        BF_2 = self.mp2(self.SL2(torch.concat([BF_1, B_2], dim=1)))

        B_3 = self.DC3_2(self.MA3_1(self.DC3_1(BF_2)))
        BF_3 = self.mp3(self.SL3(torch.concat([BF_2, B_3], dim=1)))

        B_4 = self.DC4_2(self.MA4_1(self.DC4_1(BF_3)))
        BF_4 = self.mp4(self.SL4(B_4))

        logits = self.head(BF_4)
        return logits

    def set_p(self, p):
        for i in list(self.modules()):
            if type(i) == Dist_Conv2D:
                i.p = p

    def lock_SL(self):
        for m in self.modules():
            if type(m) == Static_Layernorm:
                m.locked = True