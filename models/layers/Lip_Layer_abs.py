from os import pardir
import torch
import math
from torch import nn
import numpy as np

class Dist_Dense(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        batch_size = x.shape[0]
        x_broad = x.reshape([batch_size, self.size_in, 1])
        w_minus_x = torch.max(x_broad - self.weights, dim=1).values
        return torch.add(w_minus_x, self.bias)  # w times x + b

class Uni_Linear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        eps = 0.01

        w = self.weights
        sumw = torch.sum(torch.abs(w), dim=-1) + eps
        broadcasted_sumw = sumw.broadcast_to([self.size_in, self.size_out]).transpose(0, 1)
        w_times_x = torch.mm(x, w / broadcasted_sumw)
        return torch.add(w_times_x, self.bias)  # w times x + b

class Dist_Conv2D(nn.Module):
    torch.nn.Conv2d
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, conn_num=None):
        super().__init__()
        self.in_channels, self.out_channels, self.padding, self.kernel_size, self.stride, self.conn_num =\
            in_channels, out_channels, padding, kernel_size, stride, conn_num
        weights = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(out_channels, 1, 1)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):

        pad_l = self.padding // 2
        pad_r = self.padding - pad_l
        x_pad = nn.functional.pad(x, (pad_l, pad_r, pad_l, pad_r))

        x_unfold = x_pad.unfold(-2, self.kernel_size, self.stride).unfold(-2, self.kernel_size, self.stride)
        x_unfold = x_unfold.permute((0, 2, 3, 1, 4, 5))

        B = x_unfold.shape[0]
        Cin = self.in_channels
        Cout = self.out_channels
        H = x_unfold.shape[1]
        W = x_unfold.shape[2]
        K = self.kernel_size
        assert(Cin == self.in_channels)

        x_reshape = x_unfold.reshape((B*H*W, Cin*K*K))
        w_reshape = self.weights.reshape((Cout, Cin*K*K))

        w_diff_x = torch.cdist(x_reshape, w_reshape, p=torch.inf)

        output = w_diff_x.reshape((B, H, W, Cout)).permute((0, 3, 1, 2))

        return output + self.bias

        w = self.weights
        w_conn_shp = (w.shape[0], 1, 1) + w.shape[1:]
        w = w.reshape(w_conn_shp)

        w_diff_x = torch.abs(w - x_unfold)
        if self.conn_bound is not None:
            conn = torch.tensor(self.conn).reshape(w_conn_shp).to(device=w_diff_x.device)
            w_diff_x *= conn

        w_diff_x = torch.amax(w_diff_x, (-3, -2, -1))
        return torch.add(w_diff_x, self.bias)  # w times x + b