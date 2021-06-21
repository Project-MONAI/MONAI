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

from typing import Callable, Sequence

import torch

from monai.utils import ensure_tuple, ensure_tuple_rep

__all__ = ["generate_param_groups"]


def generate_param_groups(
    network: torch.nn.Module,
    layer_matches: Sequence[Callable],
    match_types: Sequence[str],
    lr_values: Sequence[float],
    include_others: bool = True,
):
    """
    Utility function to generate parameter groups with different LR values for optimizer.
    The output parameter groups have the same order as `layer_match` functions.

    Args:
        network: source network to generate parameter groups from.
        layer_matches: a list of callable functions to select or filter out network layer groups,
            for "select" type, the input will be the `network`, for "filter" type,
            the input will be every item of `network.named_parameters()`.
            for "select", the parameters will be
            `select_func(network).parameters()`.
            for "filter", the parameters will be
            `map(lambda x: x[1], filter(filter_func, network.named_parameters()))`
        match_types: a list of tags to identify the matching type corresponding to the `layer_matches` functions,
            can be "select" or "filter".
        lr_values: a list of LR values corresponding to the `layer_matches` functions.
        include_others: whether to include the rest layers as the last group, default to True.

    It's mainly used to set different LR values for different network elements, for example:

    .. code-block:: python

        net = Unet(dimensions=3, in_channels=1, out_channels=3, channels=[2, 2, 2], strides=[1, 1, 1])
        print(net)  # print out network components to select expected items
        print(net.named_parameters())  # print out all the named parameters to filter out expected items
        params = generate_param_groups(
            network=net,
            layer_matches=[lambda x: x.model[0], lambda x: "2.0.conv" in x[0]],
            match_types=["select", "filter"],
            lr_values=[1e-2, 1e-3],
        )
        # the groups will be a list of dictionaries:
        # [{'params': <generator object Module.parameters at 0x7f9090a70bf8>, 'lr': 0.01},
        #  {'params': <filter object at 0x7f9088fd0dd8>, 'lr': 0.001},
        #  {'params': <filter object at 0x7f9088fd0da0>}]
        optimizer = torch.optim.Adam(params, 1e-4)

    """
    layer_matches = ensure_tuple(layer_matches)
    match_types = ensure_tuple_rep(match_types, len(layer_matches))
    lr_values = ensure_tuple_rep(lr_values, len(layer_matches))

    def _get_select(f):
        def _select():
            return f(network).parameters()

        return _select

    def _get_filter(f):
        def _filter():
            # should eventually generate a list of network parameters
            return map(lambda x: x[1], filter(f, network.named_parameters()))

        return _filter

    params = []
    _layers = []
    for func, ty, lr in zip(layer_matches, match_types, lr_values):
        if ty.lower() == "select":
            layer_params = _get_select(func)
        elif ty.lower() == "filter":
            layer_params = _get_filter(func)
        else:
            raise ValueError(f"unsupported layer match type: {ty}.")

        params.append({"params": layer_params(), "lr": lr})
        _layers.extend(list(map(id, layer_params())))

    if include_others:
        params.append({"params": filter(lambda p: id(p) not in _layers, network.parameters())})

    return params
