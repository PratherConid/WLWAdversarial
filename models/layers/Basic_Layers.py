import torch
from torch import nn

class MovingAverage(nn.Module):

    def __init__(self, gamma=0.99, lock_cnt=torch.inf, ax=None):
        super().__init__()
        assert gamma > 0 and gamma < 1
        self.std = 0
        self.avg = None
        self.cnt = 0
        self.locked = False
        self.gamma = gamma
        self.lock_cnt = lock_cnt
        self.ax = ax
        self.std = 1

    def forward(self, x):
        if self.ax is not None:
            avg = torch.mean(x, axis=self.ax, keepdim=True)
        else:
            avg = torch.mean(x)

        self.cnt += 1
        if self.cnt > self.lock_cnt:
            self.locked = True
        
        if self.locked:
            return x - self.avg
        else:
            if self.avg is None:
                self.avg = ((1 - self.gamma) * avg).detach()
            else:
                self.avg = ((1 - self.gamma) * avg + self.gamma * self.avg).detach()
            return x - avg

class Static_Layernorm(nn.Module):

    def __init__(self, gamma=0.99, lock_cnt=torch.inf, ax=None):
        super().__init__()
        assert gamma > 0 and gamma < 1
        self.std = 0
        self.avg = None
        self.cnt = 0
        self.locked = False
        self.gamma = gamma
        self.lock_cnt = lock_cnt
        self.ax = ax

    def forward(self, x):
        x_dim = len(x.shape)
        var_dims = list(range(1, x_dim))
        std = torch.sqrt(torch.mean(torch.var(x, dim=var_dims)))
        if self.ax is not None:
            avg = torch.mean(x, axis=self.ax, keepdim=True)
        else:
            avg = torch.mean(x)

        self.cnt += 1
        if self.cnt > self.lock_cnt:
            self.locked = True
        if self.locked:
            return (x - self.avg) / self.std
        else:
            self.std = ((1 - self.gamma) * std + self.gamma * self.std).detach()
            if self.avg is None:
                self.avg = ((1 - self.gamma) * avg).detach()
            else:
                self.avg = ((1 - self.gamma) * avg + self.gamma * self.avg).detach()
            return (x - avg) / std