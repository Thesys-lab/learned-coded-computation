import torch
from torch import nn

from coders.coder import Coder


class MLPCoder(Coder):
    def __init__(self, num_in, num_out, in_dim, layer_sizes_multiplier):
        super().__init__(num_in, num_out, in_dim)
        self.out_dim = in_dim

        nn_modules = nn.ModuleList()
        prev_size = num_in * in_dim
        for i, size in enumerate(layer_sizes_multiplier):
            my_size = in_dim * size
            l = nn.Linear(prev_size, my_size)
            prev_size = my_size
            nn_modules.append(l)
            nn_modules.append(nn.ReLU())

        nn_modules.append(nn.Linear(prev_size, self.num_out * self.out_dim))
        self.nn = nn.Sequential(*nn_modules)

    def forward(self, in_data):
        # Flatten inputs
        val = in_data.view(in_data.size(0), -1)
        out = self.nn(val)
        return out.view(out.size(0), self.num_out, self.out_dim)


class MLPEncoder(MLPCoder):
    def __init__(self, ec_k, ec_r, in_dim):
        num_in = ec_k
        num_out = ec_r
        layer_sizes_multiplier = [ec_k]
        super().__init__(num_in, num_out, in_dim, layer_sizes_multiplier)


class MLPDecoder(MLPCoder):
    def __init__(self, ec_k, ec_r, in_dim):
        num_in = ec_k + ec_r
        num_out = ec_k
        layer_sizes_multiplier = [num_in, num_out]
        super().__init__(num_in, num_out, in_dim, layer_sizes_multiplier)
