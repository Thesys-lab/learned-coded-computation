import math
import torch
from torch import nn

from coders.coder import Coder


class ConvEncoder(Coder):
    def __init__(self, ec_k, ec_r, in_dim, intermediate_channels=20):
        super().__init__(ec_k, ec_r, in_dim)
        self.ec_k = ec_k
        self.ec_r = ec_r

        self.act = nn.ReLU()
        dim = int(math.sqrt(in_dim))
        assert dim ** 2 == in_dim, "ConvEncoder in_dim is not a square"
        self.dimensions = (dim, dim)
        int_channels = intermediate_channels * ec_k

        self.nn = nn.Sequential(
            nn.Conv2d(in_channels=self.ec_k, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=int_channels,
                      kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=self.ec_r,
                      kernel_size=1, stride=1, padding=0, dilation=1)
        )

    def forward(self, in_data):
        val = in_data.view(-1, self.ec_k,
                           self.dimensions[0], self.dimensions[1])

        out = self.nn(val)
        out = out.view(val.size(0), self.ec_r, -1)
        return out
