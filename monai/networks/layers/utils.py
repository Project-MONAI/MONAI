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

from typing import Optional, Tuple, Union

import torch.nn

from monai.networks.layers.factories import Act, Dropout, Norm, Pool, split_args
from monai.utils import has_option

__all__ = ["get_norm_layer", "get_act_layer", "get_dropout_layer", "get_pool_layer"]


def get_norm_layer(name: Union[Tuple, str], spatial_dims: Optional[int] = 1, channels: Optional[int] = 1):
    """
    Create a normalization layer instance.

    For example, to create normalization layers:

    .. code-block:: python

        from monai.networks.layers import get_norm_layer

        g_layer = get_norm_layer(name=("group", {"num_groups": 1}))
        n_layer = get_norm_layer(name="instance", spatial_dims=2)

    Args:
        name: a normalization type string or a tuple of type string and parameters.
        spatial_dims: number of spatial dimensions of the input.
        channels: number of features/channels when the normalization layer requires this parameter
            but it is not specified in the norm parameters.
    """
    if name == "":
        return torch.nn.Identity()
    norm_name, norm_args = split_args(name)
    norm_type = Norm[norm_name, spatial_dims]
    kw_args = dict(norm_args)
    if has_option(norm_type, "num_features") and "num_features" not in kw_args:
        kw_args["num_features"] = channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
        kw_args["num_channels"] = channels
    return norm_type(**kw_args)


def get_act_layer(name: Union[Tuple, str]):
    """
    Create an activation layer instance.

    For example, to create activation layers:

    .. code-block:: python

        from monai.networks.layers import get_act_layer

        s_layer = get_act_layer(name="swish")
        p_layer = get_act_layer(name=("prelu", {"num_parameters": 1, "init": 0.25}))

    Args:
        name: an activation type string or a tuple of type string and parameters.
    """
    if name == "":
        return torch.nn.Identity()
    act_name, act_args = split_args(name)
    act_type = Act[act_name]
    return act_type(**act_args)


def get_dropout_layer(name: Union[Tuple, str, float, int], dropout_dim: Optional[int] = 1):
    """
    Create a dropout layer instance.

    For example, to create dropout layers:

    .. code-block:: python

        from monai.networks.layers import get_dropout_layer

        d_layer = get_dropout_layer(name="dropout")
        a_layer = get_dropout_layer(name=("alphadropout", {"p": 0.25}))

    Args:
        name: a dropout ratio or a tuple of dropout type and parameters.
        dropout_dim: the spatial dimension of the dropout operation.
    """
    if name == "":
        return torch.nn.Identity()
    if isinstance(name, (int, float)):
        # if dropout was specified simply as a p value, use default name and make a keyword map with the value
        drop_name = Dropout.DROPOUT
        drop_args = {"p": float(name)}
    else:
        drop_name, drop_args = split_args(name)
    drop_type = Dropout[drop_name, dropout_dim]
    return drop_type(**drop_args)


def get_pool_layer(name: Union[Tuple, str], spatial_dims: Optional[int] = 1):
    """
    Create a pooling layer instance.

    For example, to create adaptiveavg layer:

    .. code-block:: python

        from monai.networks.layers import get_pool_layer

        pool_layer = get_pool_layer(("adaptiveavg", {"output_size": (1, 1, 1)}), spatial_dims=3)

    Args:
        name: a pooling type string or a tuple of type string and parameters.
        spatial_dims: number of spatial dimensions of the input.

    """
    if name == "":
        return torch.nn.Identity()
    pool_name, pool_args = split_args(name)
    pool_type = Pool[pool_name, spatial_dims]
    return pool_type(**pool_args)
