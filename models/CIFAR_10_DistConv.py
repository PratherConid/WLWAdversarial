import torch
from torch import nn
from .layers.Lip_Layer import Dist_Conv2D, Minimax_Conv2D, Uni_Linear

class CIFAR_10_DistConv(nn.Module):
    def __init__(self, p=torch.inf):
        super(CIFAR_10_DistConv, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_cnn_stack = nn.Sequential(
            Dist_Conv2D(3, 256, padding=2, p=p, conn_num=3),
            Dist_Conv2D(256, 256, padding=2, p=p, conn_num=3), nn.LayerNorm((256, 32, 32)),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(256, 512, padding=2, p=p, conn_num=3),
            Dist_Conv2D(512, 512, padding=2, p=p, conn_num=3), nn.LayerNorm((512, 16, 16)),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(512, 1024, padding=2, p=p, conn_num=3),
            Dist_Conv2D(1024, 1024, padding=2, p=p, conn_num=3), nn.LayerNorm((1024, 8, 8)),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(1024, 1024, padding=2, p=p, conn_num=3),
            Dist_Conv2D(1024, 1024, padding=2, p=p, conn_num=3), nn.LayerNorm((1024, 4, 4)),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            Uni_Linear(4096, 128), nn.LayerNorm(128), nn.ReLU(),
            Uni_Linear(128, 10), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        logits = self.linear_cnn_stack(x)
        return logits

    def set_p(self, p):
        for i in list(self.linear_cnn_stack.children()):
            if type(i) == Dist_Conv2D:
                i.p = p


class CIFAR_10_Minimax(nn.Module):
    def __init__(self):
        super(CIFAR_10_Minimax, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_cnn_stack = nn.Sequential(
            Minimax_Conv2D(3, 256, padding=2, branch=3), nn.LayerNorm((256, 32, 32)),
            nn.MaxPool2d(2, 2),
            Minimax_Conv2D(256, 512, padding=2, branch=3), nn.LayerNorm((512, 16, 16)),
            nn.MaxPool2d(2, 2),
            Minimax_Conv2D(512, 1024, padding=2, branch=3), nn.LayerNorm((1024, 8, 8)),
            nn.MaxPool2d(2, 2),
            Minimax_Conv2D(1024, 1024, padding=2, branch=3), nn.LayerNorm((1024, 4, 4)),
            nn.MaxPool2d(2, 2),
            Minimax_Conv2D(1024, 1024, padding=2, branch=3), nn.LayerNorm((1024, 2, 2)),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            Uni_Linear(1024, 128), nn.LayerNorm(128), nn.ReLU(),
            Uni_Linear(128, 10), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        logits = self.linear_cnn_stack(x)
        return logits

    def set_p(self, p):
        for i in list(self.linear_cnn_stack.children()):
            if type(i) == Dist_Conv2D:
                i.p = p