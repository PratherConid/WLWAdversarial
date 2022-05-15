import sys
import torch
from torch import nn
from dataloader import train_dataloader, test_dataloader

import sys
import os
BASE_DIR = "\\".join(os.path.abspath(__file__).split('\\')[:-4])
print(BASE_DIR)
sys.path.append(BASE_DIR)
from models.CIFAR_10_DistNets import CIFAR_10_Dis1Dis3
from model_property import static_layernorm_lip_const, count_parameters
from train import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



model = CIFAR_10_Dis1Dis3(3).to(device)
print(model)
print("Number of parameters:", count_parameters(model))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 30

trd = train_dataloader(64)
tsd = test_dataloader(64)

for t in range(epochs):
    if t >= 2:
        model.lock_SL()
    print(f"Epoch {t+1}\n-------------------------------")
    train(trd, model, loss_fn, optimizer, lip_loss_coef=0.002)
    test(tsd, model, loss_fn)
    print("Lip Const:", static_layernorm_lip_const(model))
    if t % 6 == 3:
        torch.save(model.state_dict(), "experiments/CIFAR-10/Dis1Dis3/epoch_" + str(t) + ".pt")
print("Done!")

## run: 
## cd <path>/WLWAdversarial
## nohup stdbuf -oL python -u experiments/CIFAR-10/Dis1Dis3/main.py >> experiments/CIFAR-10/Dis1Dis3/result.txt &