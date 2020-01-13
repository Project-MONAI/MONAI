import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    """Concats the forward pass input with the result from the given submodule."""

    def __init__(self, submodule, cat_dim=1):
        super().__init__()
        self.submodule = submodule
        self.cat_dim = cat_dim

    def forward(self, x):
        return torch.cat([x, self.submodule(x)], self.cat_dim)


class Flatten(nn.Module):
    """Flattens the given input in the forward pass to be [B,-1] in shape."""

    def forward(self, x):
        return x.view(x.size(0), -1)
