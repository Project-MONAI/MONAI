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

from typing import Dict, List, Optional

import torch
from torch import Tensor

from monai.inferers import SlidingWindowInferer


def ensure_dict_value_to_list_(head_outputs: Dict[str, List[Tensor]], keys: Optional[List[str]] = None) -> None:
    """
    An in-place function. We expect ``head_outputs`` to be Dict[str, List[Tensor]].
    Yet if it is Dict[str, Tensor], this func converts it to Dict[str, List[Tensor]].
    It will be modified in-place.

    Args:
        head_outputs: a Dict[str, List[Tensor]] or Dict[str, Tensor], will be modifier in-place
        keys: the keys in head_output that need to have value type List[Tensor]. If not provided, will use head_outputs.keys().
    """
    if keys is None:
        keys = list(head_outputs.keys())

    for k in keys:
        value_k = head_outputs[k]  # Tensor or List[Tensor]
        # convert value_k to List[Tensor]
        if isinstance(value_k, Tensor):
            head_outputs[k] = [value_k]
        elif isinstance(value_k[0], Tensor):
            head_outputs[k] = list(value_k)
        else:
            raise ValueError("The output of network should be Dict[str, List[Tensor]] or Dict[str, Tensor].")

    return


def check_dict_values_same_length(head_outputs: Dict[str, List[Tensor]], keys: Optional[List[str]] = None) -> None:
    """
    We expect the values in ``head_outputs``: Dict[str, List[Tensor]] to have the same length.
    Will raise ValueError if not.

    Args:
        head_outputs: a Dict[str, List[Tensor]] or Dict[str, Tensor]
        keys: the keys in head_output that need to have values (List) with same length.
            If not provided, will use head_outputs.keys().
    """
    if keys is None:
        keys = list(head_outputs.keys())

    num_output_levels_list: List[int] = []
    for k in keys:
        num_output_levels_list.append(len(head_outputs[k]))

    num_output_levels = torch.unique(torch.tensor(num_output_levels_list))
    if len(num_output_levels) != 1:
        raise ValueError(f"The values in the input dict should have the same length, Got {num_output_levels_list}.")

    return


def _network_sequence_output(images: Tensor, network, keys: Optional[List[str]] = None) -> List[Tensor]:
    """
    Decompose the output of network (a dict) into a list.

    Args:
        images: input of the network
        keys: the keys in the network output whose values will be output in this func.
            If not provided, will use all keys.

    Return:
        network output values concat to a single List[Tensor]
    """
    head_outputs = network(images)
    ensure_dict_value_to_list_(head_outputs, keys)
    if keys is None:
        keys = list(head_outputs.keys())
    check_dict_values_same_length(head_outputs, keys)
    head_outputs_sequence = []
    for _, k in enumerate(keys):
        head_outputs_sequence += list(head_outputs[k])
    return head_outputs_sequence


def predict_with_inferer(
    images: Tensor, network, keys: List[str], inferer: Optional[SlidingWindowInferer] = None
) -> Dict[str, List[Tensor]]:
    """
    Predict network dict output with an inferer. Compared with directly output network(images),
    it enables a sliding window inferer that can be used to handle large inputs.

    Args:
        images: input of the network, Tensor sized (B, C, H, W) or  (B, C, H, W, D)
        network: a network that takes an image Tensor sized (B, C, H, W) or (B, C, H, W, D) as input
            and outputs a dictionary Dict[str, List[Tensor]] or Dict[str, Tensor].
        keys: the keys in the output dict, should be network output keys or a subset of them.
        inferer: a SlidingWindowInferer to handle large inputs.

    Return:
        The predicted head_output from network, a Dict[str, List[Tensor]]

    Example:
        .. code-block:: python

            # define a naive network
            import torch
            import monai
            class NaiveNet(torch.nn.Module):
                def __init__(self, ):
                    super().__init__()

                def forward(self, images: torch.Tensor):
                    return {"cls": torch.randn(images.shape), "box_reg": [torch.randn(images.shape)]}

            # create a predictor
            network = NaiveNet()
            inferer = monai.inferers.SlidingWindowInferer(
                roi_size = (128, 128, 128),
                overlap = 0.25,
                cache_roi_weight_map = True,
            )
            network_output_keys=["cls", "box_reg"]
            images = torch.randn((2, 3, 512, 512, 512))  # a large input
            head_outputs = predict_with_inferer(images, network, network_output_keys, inferer)

    """
    if inferer is None:
        raise ValueError("Please set inferer as a monai.inferers.inferer.SlidingWindowInferer(*)")
    head_outputs_sequence = inferer(images, _network_sequence_output, network, keys=keys)
    num_output_levels: int = len(head_outputs_sequence) // len(keys)
    head_outputs = {}
    for i, k in enumerate(keys):
        head_outputs[k] = list(head_outputs_sequence[num_output_levels * i : num_output_levels * (i + 1)])
    return head_outputs
