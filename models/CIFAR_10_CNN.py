import torch
from torch import nn

# Define model
class CIFAR_10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_10_CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_cnn_stack = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, (3, 3), padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2048, 128), nn.ReLU(),
            nn.Linear(128, 10), nn.Softmax()
        )

    def forward(self, x):
        logits = self.linear_cnn_stack(x)
        return logits

class CIFAR_10_CNN_LayerNorm(nn.Module):
    def __init__(self):
        super(CIFAR_10_CNN_LayerNorm, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_cnn_stack = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1), nn.LayerNorm((64, 32, 32)), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1), nn.LayerNorm((128, 16, 16)), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1), nn.LayerNorm((256, 8, 8)), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1), nn.LayerNorm((512, 4, 4)), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2048, 128), nn.ReLU(),
            nn.Linear(128, 10), nn.Softmax()
        )

    def forward(self, x):
        logits = self.linear_cnn_stack(x)
        return logits