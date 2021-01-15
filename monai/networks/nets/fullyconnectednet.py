# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks import ADN
from monai.networks.layers.factories import Act

__all__ = ["FullyConnectedNet", "VarFullyConnectedNet"]


def _get_adn_layer(
    act: Optional[Union[Tuple, str]], dropout: Optional[Union[Tuple, str, float]], ordering: Optional[str]
) -> ADN:
    if ordering:
        return ADN(act=act, dropout=dropout, dropout_dim=1, ordering=ordering)
    return ADN(act=act, dropout=dropout, dropout_dim=1)


class FullyConnectedNet(nn.Sequential):
    """
    Plain full-connected layer neural network

    The network uses dropout and, by default, PReLU activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Sequence[int],
        dropout: Optional[Union[Tuple, str, float]] = None,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        bias: bool = True,
        adn_ordering: Optional[str] = None,
    ) -> None:
        """
        Defines a network accept input with `in_channels` channels, output of `out_channels` channels, and hidden layers
        with channels given in `hidden_channels`. If `bias` is True then linear units have a bias term.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = list(hidden_channels)
        self.add_module("flatten", nn.Flatten())
        self.adn_layer = _get_adn_layer(act, dropout, adn_ordering)

        prev_channels = self.in_channels
        for i, c in enumerate(hidden_channels):
            self.add_module("hidden_%i" % i, self._get_layer(prev_channels, c, bias))
            prev_channels = c

        self.add_module("output", nn.Linear(prev_channels, out_channels, bias))

    def _get_layer(self, in_channels: int, out_channels: int, bias: bool) -> nn.Sequential:
        seq = nn.Sequential(nn.Linear(in_channels, out_channels, bias))
        seq.add_module("ADN", self.adn_layer)
        return seq


class VarFullyConnectedNet(nn.Module):
    """Variational fully-connected network."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_size: int,
        encode_channels: Sequence[int],
        decode_channels: Sequence[int],
        dropout: Optional[Union[Tuple, str, float]] = None,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        bias: bool = True,
        adn_ordering: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size

        self.encode = nn.Sequential()
        self.decode = nn.Sequential()
        self.flatten = nn.Flatten()

        self.adn_layer = _get_adn_layer(act, dropout, adn_ordering)

        prev_channels = self.in_channels
        for i, c in enumerate(encode_channels):
            self.encode.add_module("encode_%i" % i, self._get_layer(prev_channels, c, bias))
            prev_channels = c

        self.mu = nn.Linear(prev_channels, self.latent_size)
        self.logvar = nn.Linear(prev_channels, self.latent_size)
        self.decodeL = nn.Linear(self.latent_size, prev_channels)

        for i, c in enumerate(decode_channels):
            self.decode.add_module("decode%i" % i, self._get_layer(prev_channels, c, bias))
            prev_channels = c

        self.decode.add_module("final", nn.Linear(prev_channels, out_channels, bias))

    def _get_layer(self, in_channels: int, out_channels: int, bias: bool) -> nn.Sequential:
        seq = nn.Sequential(nn.Linear(in_channels, out_channels, bias))
        seq.add_module("ADN", self.adn_layer)
        return seq

    def encode_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(x)
        x = self.flatten(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def decode_forward(self, z: torch.Tensor, use_sigmoid: bool = True) -> torch.Tensor:
        x: torch.Tensor
        x = self.decodeL(z)
        x = torch.relu(x)
        x = self.flatten(x)
        x = self.decode(x)
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)

        if self.training:  # multiply random noise with std only during training
            std = torch.randn_like(std).mul(std)

        return std.add_(mu)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_forward(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_forward(z), mu, logvar, z
