import torch
import math
from torch import nn
import numpy as np

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

class Dist_Conv2D_Dense(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride =\
            in_channels, out_channels, kernel_size, stride
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
        pad_l = (self.kernel_size[0] - 1) // 2
        pad_r = self.kernel_size[0] - 1 - pad_l
        pad_t = (self.kernel_size[1] - 1) // 2
        pad_b = self.kernel_size[1] - 1 - pad_t
        x_pad = nn.functional.pad(x, (pad_t, pad_b, pad_l, pad_r), "replicate")

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

class Dist_Conv2D_1(nn.Module):
    # random sample from input
    def __init__(self, in_channels, out_channels, conn_num):
        super().__init__()
        self.in_channels, self.out_channels, self.conn_num =\
            in_channels, out_channels, conn_num
        self.conn = np.random.randint(0, self.in_channels, self.out_channels * self.conn_num).astype(np.int64)
        self.conn = torch.from_numpy(self.conn)
        weights = torch.Tensor(out_channels, self.conn_num)
        self.weights = nn.Parameter(weights)
        # self.conn.shape = (self.out_channels * self.conn_num,)
        bias = torch.Tensor(out_channels, 1, 1)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
    
    def forward(self, x):
        # x.shape = (batch_size, C_in, W, H)
        x_conn = x[:, self.conn, :, :]
        x_conn = torch.permute(x_conn, [0, 2, 3, 1])
        # x_conn.shape = (batch_size, W, H, C_out * conn_num)
        x_conn = x_conn.reshape(x_conn.shape[:3] + (self.out_channels, self.conn_num))
        w_diff_x = torch.amax(torch.abs(self.weights - x_conn), dim=-1)
        # w_diff_x.shape = (batch_size, W, H, C_out)
        w_diff_x = torch.permute(w_diff_x, [0, 3, 1, 2])
        return torch.add(w_diff_x, self.bias)

class Dist_Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, extra_pad=0, p=torch.inf, conn_num=3):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.extra_pad, self.conn_num, self.p =\
            in_channels, out_channels, kernel_size, stride, extra_pad, conn_num, p
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
        pad_l = (self.kernel_size[0] + self.extra_pad - 1) // 2
        pad_r = self.kernel_size[0] + self.extra_pad - 1 - pad_l
        pad_t = (self.kernel_size[1] + self.extra_pad - 1) // 2
        pad_b = self.kernel_size[1] + self.extra_pad - 1 - pad_t
        x_pad = nn.functional.pad(x, (pad_t, pad_b, pad_l, pad_r), "replicate")

        x_unfold = x_pad.unfold(-2, self.kernel_size[0], self.stride).unfold(-2, self.kernel_size[1], self.stride)
        x_unfold = torch.permute(x_unfold, [0, 2, 3, 1, 4, 5])
        # x_unfold.shape = (batch_size, W_out, H_out, in_channels, kernel_size, kernel_size)
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
            w_diff_x = torch.norm(w_diff_x, dim=-1, p=self.p)
        else:
            eps=0.01
            w_diff_x = torch.norm(torch.abs(w_diff_x) + eps, dim=-1, p=self.p)
        return torch.add(w_diff_x, self.bias)

class Uni_Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, extra_pad=0, p=torch.inf, conn_num=3):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.extra_pad, self.conn_num, self.p =\
            in_channels, out_channels, kernel_size, stride, extra_pad, conn_num, p
        self.accl_sz = in_channels * kernel_size[0] * kernel_size[1]
        if conn_num is not None:
            weights = torch.Tensor(out_channels, self.conn_num)
            self.conn = np.random.randint(0, self.accl_sz, (self.out_channels, self.conn_num)).astype(np.int64)
            self.conn = nn.functional.one_hot(torch.from_numpy(self.conn), num_classes=self.accl_sz)
            self.conn = self.conn.type(torch.float32)
            # self.conn.shape = (self.out_channels, self.conn_num, self.accl_sz)
        else:
            weights = torch.Tensor(self.accl_sz, out_channels)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(out_channels, 1, 1)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        pad_l = (self.kernel_size[0] + self.extra_pad - 1) // 2
        pad_r = self.kernel_size[0] + self.extra_pad - 1 - pad_l
        pad_t = (self.kernel_size[1] + self.extra_pad - 1) // 2
        pad_b = self.kernel_size[1] + self.extra_pad - 1 - pad_t
        x_pad = nn.functional.pad(x, (pad_t, pad_b, pad_l, pad_r), "replicate")

        x_unfold = x_pad.unfold(-2, self.kernel_size[0], self.stride).unfold(-2, self.kernel_size[1], self.stride)
        x_unfold = torch.permute(x_unfold, [0, 2, 3, 1, 4, 5])
        # x_unfold.shape = (batch_size, W_out, H_out, in_channels, kernel_size, kernel_size)
        if self.conn_num is not None:
            x_unfold = x_unfold.reshape(x_unfold.shape[0:3] + (self.accl_sz,))
            # x_unfold.shape = (batch_size, W_out, H_out, accl_sz)
            conn = self.conn.to(device=x.device)
            # conn.shape = (C_out, conn_num, accl_sz)
            w = torch.einsum("Cc,Cca->aC", self.weights, conn)
            # w.shape = (accl_sz, C_out)
        else:
            x_unfold = x_unfold.reshape(x_unfold.shape[0:3] + (self.accl_sz,))
            # x_unfold.shape = (batch_size, W_out, H_out, accl_sz)

        eps = 0.01
        sumw = torch.sum(torch.abs(w), dim=0) + eps
        broadcasted_sumw = sumw.broadcast_to([self.accl_sz, self.out_channels])
        wxb = torch.matmul(x_unfold, w / broadcasted_sumw)
        # wxb.shape = (batch_size, W_out, H_out, C_out)
        wxb = torch.permute(wxb, [0, 3, 1, 2])
        return torch.add(wxb, self.bias)

class Minimax_Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, extra_pad=0, branch=3, abs=True):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.extra_pad, self.branch =\
            in_channels, out_channels, kernel_size, stride, extra_pad, branch
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
        self.abs = abs

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5)) # weight init
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        pad_l = (self.kernel_size[0] + self.extra_pad - 1) // 2
        pad_r = self.kernel_size[0] + self.extra_pad - 1 - pad_l
        pad_t = (self.kernel_size[1] + self.extra_pad - 1) // 2
        pad_b = self.kernel_size[1] + self.extra_pad - 1 - pad_t
        x_pad = nn.functional.pad(x, (pad_t, pad_b, pad_l, pad_r), "replicate")

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

        if self.abs:
            w_diff_x = torch.abs(x_unfold - w1)
        else:
            w_diff_x = x_unfold - w1
        # w_diff_x.shape = (batch_size, C_out, W_out, H_out, bran * bran)

        ma = torch.amax(w_diff_x.reshape(w_diff_x.shape[:-1] + (self.branch, self.branch)), dim=-1)
        # ma.shape = (batch_size, C_out, W_out, H_out, bran)

        if self.abs:
            mi = torch.amin(torch.abs(ma - w2), dim=-1)
        else:
            mi = torch.amin(ma - w2, dim=-1)
        return mi