# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from monai.networks.layers import get_act_layer
from monai.utils import look_up_option

SUPPORTED_DROPOUT_MODE = {"vit", "swin"}


class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
        act: tuple | str = "GELU",
        dropout_mode="vit",
        output_dim: Optional[int] = None,
        num_layers: int = 2,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer. Input size.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used. Output dim.
            dropout_rate: fraction of the input units to drop.
            act: activation type and arguments. Defaults to GELU. Also supports "GEGLU" and others.
            dropout_mode: dropout mode, can be "vit" or "swin". `self.dropout2` is only applied for the last layer.
            output_dim: output tensor dimension, if `None` `hidden_size` is used as output dimension.
            num_layers: number of mlp layers
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.num_layers = num_layers
        h = [mlp_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([hidden_size] + h, h + [output_dim or hidden_size]))
        self.fn = get_act_layer(act)
        self.drop1 = nn.Dropout(dropout_rate)
        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.fn(layer(x)) if i < self.num_layers - 1 else layer(x)
            drop = self.drop1 if i < self.num_layers - 1 else self.drop2
            x = drop(x)
        return x
