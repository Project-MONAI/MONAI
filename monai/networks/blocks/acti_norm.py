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

from typing import Optional, Tuple, Union

import torch.nn as nn

from monai.networks.layers.factories import Act, Dropout, Norm, split_args
from monai.utils import has_option


class ADN(nn.Sequential):
    """
    Constructs a sequential module of optional activation, dropout, and normalization layers
    (with an arbitrary order)::

        -- (Norm) -- (Dropout) -- (Acti) --

    Args:
        ordering: a string representing the ordering of activation, dropout, and normalization. Defaults to "NDA".
        in_channels: `C` from an expected input of size (N, C, H[, W, D]).
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        norm_dim: determine the spatial dimensions of the normalization layer.
            defaults to `dropout_dim` if unspecified.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout.
            defaults to `norm_dim` if unspecified.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

    Examples::

        # activation, group norm, dropout
        >>> norm_params = ("GROUP", {"num_groups": 1, "affine": False})
        >>> ADN(norm=norm_params, in_channels=1, dropout_dim=1, dropout=0.8, ordering="AND")
        ADN(
            (A): ReLU()
            (N): GroupNorm(1, 1, eps=1e-05, affine=False)
            (D): Dropout(p=0.8, inplace=False)
        )

        # LeakyReLU, dropout
        >>> act_params = ("leakyrelu", {"negative_slope": 0.1, "inplace": True})
        >>> ADN(act=act_params, in_channels=1, dropout_dim=1, dropout=0.8, ordering="AD")
        ADN(
            (A): LeakyReLU(negative_slope=0.1, inplace=True)
            (D): Dropout(p=0.8, inplace=False)
        )

    See also:

        :py:class:`monai.networks.layers.Dropout`
        :py:class:`monai.networks.layers.Act`
        :py:class:`monai.networks.layers.Norm`
        :py:class:`monai.networks.layers.split_args`

    """

    def __init__(
        self,
        ordering: str = "NDA",
        in_channels: Optional[int] = None,
        act: Optional[Union[Tuple, str]] = "RELU",
        norm: Optional[Union[Tuple, str]] = None,
        norm_dim: Optional[int] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        op_dict = {"A": None, "D": None, "N": None}
        # define the normalization type and the arguments to the constructor
        if norm is not None:
            if norm_dim is None and dropout_dim is None:
                raise ValueError("norm_dim or dropout_dim needs to be specified.")
            norm_name, norm_args = split_args(norm)
            norm_type = Norm[norm_name, norm_dim or dropout_dim]
            kw_args = dict(norm_args)
            if has_option(norm_type, "num_features") and "num_features" not in kw_args:
                kw_args["num_features"] = in_channels
            if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
                kw_args["num_channels"] = in_channels
            op_dict["N"] = norm_type(**kw_args)

        # define the activation type and the arguments to the constructor
        if act is not None:
            act_name, act_args = split_args(act)
            act_type = Act[act_name]
            op_dict["A"] = act_type(**act_args)

        if dropout is not None:
            # if dropout was specified simply as a p value, use default name and make a keyword map with the value
            if isinstance(dropout, (int, float)):
                drop_name = Dropout.DROPOUT
                drop_args = {"p": float(dropout)}
            else:
                drop_name, drop_args = split_args(dropout)

            if norm_dim is None and dropout_dim is None:
                raise ValueError("norm_dim or dropout_dim needs to be specified.")
            drop_type = Dropout[drop_name, dropout_dim or norm_dim]
            op_dict["D"] = drop_type(**drop_args)

        for item in ordering.upper():
            if item not in op_dict:
                raise ValueError(f"ordering must be a string of {op_dict}, got {item} in it.")
            if op_dict[item] is not None:
                self.add_module(item, op_dict[item])  # type: ignore
