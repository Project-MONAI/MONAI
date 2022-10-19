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

__all__ = ["BaseEncoder"]


class BaseEncoder(metaclass=ABCMeta):
    """
    Abstract class defines interface of encoders in flexible unet.
    Encoders in flexible unet must derive from this class. Each interface method
    should return a list containing relative information about a series of newtworks
    defined by encoder. For example, the efficient-net encoder implement 10 basic
    network structures in one encoder. When calling `get_encoder_name_string_list`
    function, a string list like ["efficientnet-b0", "efficientnet-b1" ... "efficientnet-l2"]
    should be returned.
    """

    @classmethod
    @abstractmethod
    def get_encoder_parameters(cls) -> List[Dict]:
        """
        Get parameter list to initialize encoder networks.
        Each parameter dict must have `spatial_dims`, `in_channels`
        and `pretrained` parameters.
        The reason that this function should return a list is that a
        series of encoders can be implemented by one encoder class
        given different initialization parameters. Each parameter dict
        in return list should be able to initialize a unique encoder.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_output_feature_channels(cls) -> List[Tuple[int, ...]]:
        """
        Get number of output features' channels.
        The reason that this function should return a list is that a
        series of encoders can be implemented by one encoder class
        given different initialization parameters. And it is possible
        that different encoders have different output feature map
        channels. Therefore a list of output feature map channel tuples
        corresponding to each encoder should be returned by this method.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_output_feature_numbers(cls) -> List[int]:
        """
        Get number of output feature.
        The reason that this function should return a list is that a
        series of encoders can be implemented by one encoder class
        given different initialization parameters. And it is possible
        that different encoders have different output feature numbers.
        Therefore a list of output feature numbers corresponding to
        each encoder should be returned by this method.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_encoder_names(cls) -> List[str]:
        """
        Get the name string of encoders which will be used to initialize
        flexible unet.
        The reason that this function should return a list is that a
        series of encoders can be implemented by one encoder class
        given different initialization parameters. And a name string is
        the key to each encoder in flexible unet backbone registry.
        Therefore this method should return every encoder name that needs
        to be registed in flexible unet.
        """
        raise NotImplementedError
