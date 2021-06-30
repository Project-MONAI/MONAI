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


import math
from typing import Tuple, Union

import torch
import torch.nn as nn

from monai.utils import optional_import

einops, has_einops = optional_import("einops")


class PatchEmbeddingBlock(nn.Module):
    """
    A patch embedding block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Union[int, Tuple[int, int, int]],
        patch_size: Union[int, Tuple[int, int, int]],
        hidden_size: int,
        num_heads: int,
        pos_embed: Union[Tuple, str], # type: ignore
        classification: bool,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if img_size < patch_size:
            raise AssertionError("patch_size should be smaller than img_size.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        if pos_embed == "perceptron":
            if img_size[0] % patch_size[0] != 0:
                raise AssertionError("img_size should be divisible by patch_size for perceptron patch embedding.")

        if has_einops:
            from einops.layers.torch import Rearrange

            self.Rearrange = Rearrange
        else:
            raise ValueError('"Requires einops.')

        self.n_patches = (
            (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])  # type: ignore
        )
        self.patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.pos_embed = pos_embed
        if self.pos_embed == "conv":
            self.patch_embeddings = nn.Conv3d(
                in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size  # type: ignore
            )
        elif self.pos_embed == "perceptron":
            self.patch_embeddings = nn.Sequential(  # type: ignore
                self.Rearrange(
                    "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)",
                    p1=patch_size[0],
                    p2=patch_size[1],
                    p3=patch_size[2],
                ),
                nn.Linear(self.patch_dim, hidden_size),
            )
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
        self.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def trunc_normal_(self, tensor, mean, std, a, b):
        # From PyTorch official master until it's in a few official releases - RW
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        with torch.no_grad():
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)
            tensor.uniform_(2 * l - 1, 2 * u - 1)
            tensor.erfinv_()
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)
            tensor.clamp_(min=a, max=b)
            return tensor

    def forward(self, x):
        if self.pos_embed == "conv":
            x = self.patch_embeddings(x)
            x = x.flatten(2)
            x = x.transpose(-1, -2)
        elif self.pos_embed == "perceptron":
            x = self.patch_embeddings(x)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
