# Copyright 2020 MONAI Consortium
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


def generate_params(network: torch.nn.Module, layer_matches: Sequence[Callable], lr_values: Sequence[float]):
    """
    Utility function to generate parameter groups with different LR values for optimizer.

    Args:
        layer_matches: a list of callable functions to select network layer groups, input will be the network.
        lr_values: a list of LR values corresponding to the `layer_matches` functions.

    It's mainly used to set different init LR values for different network elements, for example::

        net = Unet(dimensions=3, in_channels=1, out_channels=3).to(device)
        print(net)  # print out network components to select expected items
        params = generate_params(net, [lambda x: x.model[-1], lambda x: x.model[-2]], [1e-2, 1e-3])
        optimizer = torch.optim.Adam(params, 1e-4)

    """
    if len(layer_matches) != len(lr_values):
        raise ValueError("length of layer_match callable functions and LR values should be the same.")

    params = list()
    _layers = list()
    for func, lr in zip(layer_matches, lr_values):
        layer_params = func(network).parameters()
        params.append({"params": layer_params, "lr": lr})
        _layers.extend(list(map(id, layer_params)))
    params.append({"params": filter(lambda p: id(p) not in _layers, network.parameters())})

    return params
