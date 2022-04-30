import torch
from torch import nn
from .layers.Lip_Layer import Dist_Conv2D, Minimax_Conv2D, Uni_Linear, Static_Layernorm

class CIFAR_10_DistConv(nn.Module):
    def __init__(self, conn_num, p=torch.inf):
        super(CIFAR_10_DistConv, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_cnn_stack = nn.Sequential(
            Dist_Conv2D(3, 256, padding=2, p=p, conn_num=conn_num),
            Dist_Conv2D(256, 256, padding=2, p=p, conn_num=conn_num), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(256, 1024, padding=2, p=p, conn_num=conn_num),
            Dist_Conv2D(1024, 1024, padding=2, p=p, conn_num=conn_num), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(1024, 4096, padding=2, p=p, conn_num=conn_num),
            Dist_Conv2D(4096, 4096, padding=2, p=p, conn_num=conn_num), Static_Layernorm(),
            Dist_Conv2D(4096, 2048, padding=2, p=p, conn_num=conn_num),
            Dist_Conv2D(2048, 1024, padding=2, p=p, conn_num=conn_num), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(1024, 256, padding=2, p=p, conn_num=8),
            Dist_Conv2D(256, 64, padding=2, p=p, conn_num=8), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            Uni_Linear(256, 80), Static_Layernorm(), nn.ReLU(),
            Uni_Linear(80, 10), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        logits = self.linear_cnn_stack(x)
        return logits

    def set_p(self, p):
        for i in list(self.linear_cnn_stack.children()):
            if type(i) == Dist_Conv2D:
                i.p = p

    def lock_SL(self):
        for m in self.linear_cnn_stack.modules():
            if type(m) == Static_Layernorm:
                m.locked = True


class CIFAR_10_Minimax(nn.Module):
    def __init__(self, conn_num, abs=True):
        super(CIFAR_10_Minimax, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_cnn_stack = nn.Sequential(
            Minimax_Conv2D(3, 512, padding=2, branch=conn_num, abs=abs), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            Minimax_Conv2D(512, 1024, padding=2, branch=conn_num, abs=abs), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            Minimax_Conv2D(1024, 1024, padding=2, branch=conn_num, abs=abs), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            Minimax_Conv2D(1024, 512, padding=2, branch=conn_num, abs=abs), Static_Layernorm(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            Uni_Linear(2048, 80), Static_Layernorm(), nn.ReLU(),
            Uni_Linear(80, 10), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        logits = self.linear_cnn_stack(x)
        return logits

    def set_p(self, p):
        for i in list(self.linear_cnn_stack.children()):
            if type(i) == Dist_Conv2D:
                i.p = p

    def lock_SL(self):
        for m in self.linear_cnn_stack.modules():
            if type(m) == Static_Layernorm:
                m.locked = True