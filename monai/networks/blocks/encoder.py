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

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple

__all__ = ["BasicEncoder"]


class BasicEncoder(metaclass=ABCMeta):
    """
    Abstract class defines inferface of backbones/encoders in flexible unet.
    The encoders/backbones in flexible unet must derive from this class. Each interface
    should return a list containing relative information about a series of newtworks
    defined by encoder. For example, the efficient-net encoder implement 10 basic
    network structures in one encoder. When calling `get_encoder_name_string_list`
    function, a string list like ["efficientnet-b0", "efficientnet-b1" ... "efficientnet-l2"]
    should be returned.
    """

    @classmethod
    @abstractmethod
    def get_backbone_parameter(cls) -> List[Dict]:
        """
        Get parameter list to initialize encoder networks.
        Each parameter dict must have `spatial_dims`, `in_channels`
        and `pretrained` parameters.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_output_feature_channel_list(cls) -> List[Tuple[int, ...]]:
        """
        Get number of output features' channel.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_output_feature_number_list(cls) -> List[int]:
        """
        Get number of output feature.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_encoder_name_string_list(cls) -> List[str]:
        """
        Get the name string of backbones which will be used to initialize flexible unet.
        """
        raise NotImplementedError
