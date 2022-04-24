import torch
import math
from torch import nn
import numpy as np

class Static_Layernorm(nn.Module):
    def __init__(self, gamma_dec=0.2, warmup=200):
        super().__init__()
        assert gamma_dec > 0 and gamma_dec < 1
        assert type(warmup) == int and warmup >= 1
        self.std = 0
        self.avg = 0
        self.cnt = 0
        self.gamma_dec = gamma_dec
        self.warmup = warmup

    def forward(self, x):
        std = torch.sqrt(torch.var(x))
        avg = torch.mean(x)
        if self.cnt < self.warmup:
            self.std = std
            self.avg = avg
            return (x - avg) / std
        self.cnt += 1
        alpha = (self.cnt - self.warmup) ** (-self.gamma_dec)
        self.std = float(alpha * std + (1 - alpha) * self.std)
        self.avg = float(alpha * avg + (1 - alpha) * self.avg)
        return (x - self.avg) / self.std

class Dist_Dense(nn.Module):
    def __init__(self, size_in, size_out, p=torch.inf):
        super().__init__()
        self.size_in, self.size_out, self.p = size_in, size_out, p
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
        # x.shape = (batch_size, size_in)
        w_minus_x = torch.cdist(x, self.weights.to(device=x.device), p=self.p)
        return torch.add(w_minus_x, self.bias.to(device=x.device))  # w times x + b

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
        broadcasted_sumw = sumw.broadcast_to([self.size_in, self.size_out])
        w_times_x = torch.mm(x, w.t() / broadcasted_sumw)
        return torch.add(w_times_x, self.bias)  # w times x + b

class Dist_Conv2D(nn.Module):
    torch.nn.Conv2d
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, p=torch.inf, conn_num=None):
        super().__init__()
        self.in_channels, self.out_channels, self.padding, self.kernel_size, self.stride, self.conn_num, self.p =\
            in_channels, out_channels, padding, kernel_size, stride, conn_num, p
        self.accl_sz = in_channels * kernel_size[0] * kernel_size[1]
        if conn_num is not None:
            weights = torch.Tensor(out_channels, self.conn_num)
            self.conn = np.random.randint(0, self.accl_sz, self.out_channels * self.conn_num).astype(np.int64)
            self.conn = torch.from_numpy(self.conn)
            # self.conn.shape = (self.out_channels * self.conn_num,)
        else:
            weights = torch.Tensor(out_channels, self.accl_sz)
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
        x_pad = nn.functional.pad(x, (pad_l, pad_r, pad_l, pad_r), "replicate")
        x_unfold = x_pad.unfold(-2, self.kernel_size[0], self.stride).unfold(-2, self.kernel_size[1], self.stride)
        x_unfold = torch.permute(x_unfold, [0, 2, 3, 1, 4, 5])
        if self.conn_num is not None:
            x_unfold = x_unfold.reshape(x_unfold.shape[0:3] + (self.accl_sz,))
            # x_unfold.shape = (batch_size, W_out, H_out, accl_sz)
            conn = self.conn.to(device=x.device)
            # conn.shape = (C_out * conn_num)
            x_unfold = x_unfold[..., conn]
            x_unfold = x_unfold.reshape(x_unfold.shape[:-1] + (self.out_channels, self.conn_num))
            # x_unfold.shape = (batch_size, W_out, H_out, C_out, conn_num)
            x_unfold = torch.permute(x_unfold, [0, 3, 1, 2, 4])
            # x_unfold.shape = (batch_size, C_out, W_out, H_out, conn_num)
        else:
            x_unfold = x_unfold.reshape((x_unfold.shape[0], 1) + x_unfold.shape[1:3] + (self.accl_sz,))
            # x_unfold.shape = (batch_size, 1, W_out, H_out, accl_sz)

        w = self.weights
        w_conn_shp = (w.shape[0], 1, 1) + w.shape[1:]
        w = w.reshape(w_conn_shp)
        # w.shape = (C_out, 1, 1, accl_sz / conn_num)

        w_diff_x = w - x_unfold
        # w_diff_x.shape = (batch_size, C_out, W_out, H_out, accl_size / conn_num)

        if self.p == torch.inf:
            w_diff_x = torch.amax(torch.abs(w_diff_x), dim=-1)
        elif self.p > 0:
            w_diff_x = torch.norm(w_diff_x, dim=-1, p = self.p)
        else:
            eps=0.01
            w_diff_x = torch.norm(torch.abs(w_diff_x) + eps, dim=-1, p=self.p)
        return torch.add(w_diff_x, self.bias)

class Minimax_Conv2D(nn.Module):
    torch.nn.Conv2d
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0, branch=3):
        super().__init__()
        self.in_channels, self.out_channels, self.padding, self.kernel_size, self.stride, self.branch =\
            in_channels, out_channels, padding, kernel_size, stride, branch
        self.accl_sz = in_channels * kernel_size[0] * kernel_size[1]
        self.conn = np.random.randint(0, self.accl_sz, self.out_channels * self.branch * self.branch).astype(np.int64)
        self.conn = torch.from_numpy(self.conn)
        # self.conn.shape = (self.out_channels * self.branch * self.branch,)
        w1 = torch.Tensor(out_channels, self.branch * self.branch)
        self.w1 = nn.Parameter(w1)
        w2 = torch.Tensor(out_channels, self.branch)
        self.w2 = nn.Parameter(w2)
        bias = torch.Tensor(out_channels, 1, 1)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5)) # weight init
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        pad_l = self.padding // 2
        pad_r = self.padding - pad_l
        x_pad = nn.functional.pad(x, (pad_l, pad_r, pad_l, pad_r), "replicate")
        x_unfold = x_pad.unfold(-2, self.kernel_size[0], self.stride).unfold(-2, self.kernel_size[1], self.stride)
        x_unfold = torch.permute(x_unfold, [0, 2, 3, 1, 4, 5])
        x_unfold = x_unfold.reshape(x_unfold.shape[0:3] + (self.accl_sz,))
        # x_unfold.shape = (batch_size, W_out, H_out, accl_sz)

        conn = self.conn.to(device=x.device)
        # conn.shape = (C_out * bran * bran)
        x_unfold = x_unfold[..., conn]
        x_unfold = x_unfold = x_unfold.reshape(x_unfold.shape[:-1] + (self.out_channels, self.branch * self.branch))
        # x_unfold.shape = (batch_size, W_out, H_out, C_out, bran * bran)
        x_unfold = torch.permute(x_unfold, [0, 3, 1, 2, 4])
        # x_unfold.shape = (batch_size, C_out, W_out, H_out, bran * bran)

        w1 = self.w1
        w1_shp = (w1.shape[0], 1, 1) + w1.shape[1:]
        w1 = w1.reshape(w1_shp)
        # w.shape = (C_out, 1, 1, bran * bran)

        w2 = self.w2
        w2_shp = (w2.shape[0], 1, 1) + w2.shape[1:]
        w2 = w2.reshape(w2_shp)

        w_diff_x = x_unfold - w1
        # w_diff_x.shape = (batch_size, C_out, W_out, H_out, bran * bran)

        ma = torch.amax(w_diff_x.reshape(w_diff_x.shape[:-1] + (self.branch, self.branch)), dim=-1)
        # ma.shape = (batch_size, C_out, W_out, H_out, bran)
        mi = torch.amin(ma - w2, dim=-1)
        return mi