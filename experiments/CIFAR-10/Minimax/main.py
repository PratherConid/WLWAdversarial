import sys
import torch
from torch import nn
from dataloader import train_dataloader, test_dataloader

import sys
import os
BASE_DIR = "\\".join(os.path.abspath(__file__).split('\\')[:-4])
print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.CIFAR_10_DistConv import CIFAR_10_Minimax
from train import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



model = CIFAR_10_Minimax().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 20

trd = train_dataloader(32)
tsd = test_dataloader(32)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trd, model, loss_fn, optimizer)
    test(tsd, model, loss_fn)
    if t % 4 == 3:
        torch.save(model.state_dict(), "experiments/CIFAR-10/Classical/epoch_" + str(t) + ".pt")
print("Done!")

## run: 
## cd <path>/WLWAdversarial
## nohup stdbuf -oL python -u experiments/CIFAR-10/Minimax/main.py >> experiments/CIFAR-10/Minimax/result.txt &