import sys
import torch
from torch import nn
from dataloader import train_dataloader, test_dataloader

import sys
import os
BASE_DIR = "\\".join(os.path.abspath(__file__).split('\\')[:-4])
print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.CIFAR_10_CNN import CIFAR_10_CNN_LayerNorm
from train import gradloss_train, gradient_test

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



model = CIFAR_10_CNN_LayerNorm().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 20

trd = train_dataloader(32)
tsd = test_dataloader(32)

for t in range(epochs):
    sys.stdout.flush()
    print(f"Epoch {t+1}\n-------------------------------")
    gradloss_train(train_dataloader, model, loss_fn, optimizer, mu=0.05, device=device)
    gradient_test(test_dataloader, model, loss_fn, gl_ratio=0.2, device=device)
    if t == 0 or t % 4 == 3:
        torch.save(model.state_dict(), "experiments/CIFAR-10/GradLoss/epoch_" + str(t) + ".pt")
print("Done!")

## run: 
## cd <path>/WLWAdversarial
## nohup stdbuf -oL python -u experiments/CIFAR-10/GradLoss/main.py >> experiments/CIFAR-10/GradLoss/result.txt &