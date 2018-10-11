import torch
from torch import nn


class Logistic(nn.Module):
    """
    Single linear layer. A softmax layer should be used on the output.
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, in_data):
        return self.lin(in_data)
