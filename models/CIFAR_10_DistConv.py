from torch import nn
from .layers.Lip_Layer import Dist_Conv2D, Uni_Linear

class CIFAR_10_DistConv(nn.Module):
    def __init__(self):
        super(CIFAR_10_DistConv, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_cnn_stack = nn.Sequential(
            Dist_Conv2D(3, 256, padding=2, conn_num=3),
            Dist_Conv2D(256, 256, padding=2, conn_num=3), nn.LayerNorm((256, 32, 32)),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(256, 512, padding=2, conn_num=3),
            Dist_Conv2D(512, 512, padding=2, conn_num=3), nn.LayerNorm((512, 16, 16)),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(512, 512, padding=2, conn_num=3),
            Dist_Conv2D(512, 512, padding=2, conn_num=3), nn.LayerNorm((512, 8, 8)),
            nn.MaxPool2d(2, 2),
            Dist_Conv2D(512, 512, padding=2, conn_num=3),
            Dist_Conv2D(512, 512, padding=2, conn_num=3), nn.LayerNorm((512, 4, 4)),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            Uni_Linear(2048, 128), nn.ReLU(), nn.LayerNorm(128),
            Uni_Linear(128, 10), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        logits = self.linear_cnn_stack(x)
        return logits