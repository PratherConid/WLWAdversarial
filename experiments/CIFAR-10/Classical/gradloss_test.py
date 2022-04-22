import sys
import torch
from torch import nn
from dataloader import test_dataloader

import sys
import os
BASE_DIR = "\\".join(os.path.abspath(__file__).split('\\')[:-4])
print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.CIFAR_10_CNN import CIFAR_10_CNN
from train import gradient_test

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



classical_model = CIFAR_10_CNN()
classical_model.load_state_dict(torch.load("experiments\CIFAR-10\Classical\epoch_19.pt"))
classical_model.to(device)
gradient_test(test_dataloader(16), classical_model, nn.CrossEntropyLoss(), gl_ratio=0.2, device=device)

## Result: Accuracy: 82.2%, Avg loss: 1.637881, Avg Gradient loss: 84.358486