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
from torch import Tensor, nn

from monai.inferers import SlidingWindowInferer


def convert_dict_value_to_list(head_outputs: Dict[str, List[Tensor]]) -> Dict[str, List[Tensor]]:
    """
    We expect ``head_outputs`` to be Dict[str, List[Tensor]].
    Yet if it is Dict[str, Tensor], this func converts it to Dict[str, List[Tensor]].

    Args:
        head_outputs: a Dict[str, List[Tensor]] or Dict[str, Tensor]

    Return:
        a Dict[str, List[Tensor]]
    """
    head_outputs_standard: Dict[str, List[Tensor]] = {}
    for k in head_outputs.keys():
        value_k = head_outputs[k]  # Tensor or List[Tensor]
        # convert value_k to List[Tensor]
        if isinstance(value_k, Tensor):
            head_outputs_standard[k] = [value_k]  # type: ignore
        elif isinstance(value_k[0], Tensor):
            head_outputs_standard[k] = list(value_k)
        else:
            raise ValueError("The output of network should be Dict[str, List[Tensor]] or Dict[str, Tensor].")

    return head_outputs_standard  # type: ignore


def check_dict_values_same_length(head_outputs: Dict[str, List[Tensor]]) -> None:
    """
    We expect the values in ``head_outputs``: Dict[str, List[Tensor]] to have the same length.
    Will raise ValueError if not.

    Args:
        head_outputs: a Dict[str, List[Tensor]] or Dict[str, Tensor]
    """
    num_output_levels_list: List[int] = []
    for k in head_outputs.keys():
        num_output_levels_list.append(len(head_outputs[k]))

    num_output_levels = torch.unique(torch.tensor(num_output_levels_list))
    if len(num_output_levels) != 1:
        raise ValueError(f"The values in the input dict should have the same length, Got {num_output_levels_list}.")

    return


def _network_sequence_output(images: Tensor, network, keys: List[str]) -> List[Tensor]:
    """
    Decompose the output of network (a dict) into a list.

    Args:
        images: input of the network

    Return:
        network output list
    """
    head_outputs = convert_dict_value_to_list(network(images))
    check_dict_values_same_length(head_outputs)
    head_outputs_sequence = []
    for k in keys:
        head_outputs_sequence += list(head_outputs[k])
    return head_outputs_sequence


def predict_with_inferer(
    images: Tensor, network, keys: List[str], inferer: Optional[SlidingWindowInferer] = None
) -> List[Tensor]:
    """
    Predict network dict output with an inferer. Compared with directly output network(images),
    it enables a sliding window inferer that can be used to handle large inputs.

    Args:
        images: input of the network, Tensor sized (B, C, H, W) or  (B, C, H, W, D)
        network: a network that takes an image Tensor sized (B, C, H, W) or (B, C, H, W, D) as input
            and outputs a dictionary Dict[str, List[Tensor]] or Dict[str, Tensor].
        keys: the keys in network output.
        inferer: a SlidingWindowInferer to handle large inputs.

    Example:
        .. code-block:: python

            # define a naive network
            import torch
            import monai
            class NaiveNet(torch.nn.Module):
                def __init__(self, ):
                    super().__init__()

                def forward(self, images: torch.Tensor):
                    return {"cls": [torch.randn(images.shape)], "box_reg": [torch.randn(images.shape)]}

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


class DictPredictor(nn.Module):
    """
    Predictor that works for network with Dict output. Compared with directly output self.network(images),
    this predictor enables a sliding window inferer that can be used to handle large inputs.

    The input to the predictor is expected to be a Tensor sized (B, C, H, W) or  (B, C, H, W, D).
    The output of the predictor is a Dict[str, List[Tensor]]

    Args:
        network: a network that takes an image Tensor sized (B, C, H, W) or (B, C, H, W, D) as input
            and outputs a dictionary Dict[str, List[Tensor]] or Dict[str, Tensor].
        network_output_keys: the keys in the output of the network.
        inferer: a SlidingWindowInferer to handle large inputs.

    Example:
        .. code-block:: python

            # define a naive network
            import torch
            import monai
            class NaiveNet(torch.nn.Module):
                def __init__(self, ):
                    super().__init__()

                def forward(self, images: torch.Tensor):
                    return {"cls": [torch.randn(images.shape)], "box_reg": [torch.randn(images.shape)]}

            # create a predictor
            net = NaiveNet()
            inferer = monai.inferers.SlidingWindowInferer(
                roi_size = (128, 128, 128),
                overlap = 0.25,
                cache_roi_weight_map = True,
            )
            predictor = RetinaNetPredictor(net, network_output_keys=["cls", "box_reg"], inferer=inferer)
            images = torch.randn((2, 3, 512, 512, 512))  # a large input
            head_outputs = predictor.forward_with_inferer(images)

    """

    def __init__(self, network, network_output_keys: List[str], inferer: Optional[SlidingWindowInferer] = None):
        super().__init__()
        self.network = network
        self.keys = network_output_keys
        self.inferer = inferer
        self.num_output_levels: int = 1

    def forward_with_inferer(self, images: Tensor) -> Dict[str, List[Tensor]]:
        """
        Compute the output of network using self.inferer,
        a :class:`~monai.inferers.inferer.SlidingWindowInferer`,
        to handle large input data.

        We expect the output of self.network to be Dict[str, List[Tensor]].
        Yet if it is Dict[str, Tensor], it will be converted to Dict[str, List[Tensor]].

        Args:
            images: input of the network, Tensor sized (B, C, H, W) or  (B, C, H, W, D)

        Return:
            The output of the network, Dict[str, List[Tensor]]
        """
        # if use_inferer, we need to decompose the output dict into sequence,
        # then do infererence, finally reconstruct dict.
        return predict_with_inferer(images, self.network, self.keys, self.inferer)

    def forward(self, images: Tensor) -> Dict[str, List[Tensor]]:
        """
        Compute the output of network without using any inferer.
        We expect the output of self.network to be Dict[str, List[Tensor]].
        Yet if it is Dict[str, Tensor], it will be converted to Dict[str, List[Tensor]].

        Args:
            images: input of the network, Tensor sized (B, C, H, W) or  (B, C, H, W, D)

        Return:
            The output of the network, Dict[str, List[Tensor]]
        """
        head_outputs = convert_dict_value_to_list(self.network(images))
        check_dict_values_same_length(head_outputs)
        return head_outputs
