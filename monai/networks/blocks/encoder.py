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
    An abstract class defines inferfaces of backbones/encoders in flexible unet.
    The encoders/backbones in flexible unet need to derive from this class and
    implement all the abstractmethods in this class. A list contains a series relative
    information about networks should be returned, in case the encoder defines a
    series of network structures. For example, the efficient-net encoder implement 10
    basic network structures in an encoder. When calling a _get_encoder_name_string_list
    function, a string list ["efficientnet-b0", "efficientnet-b1" ... "efficientnet-l2"]
    should be returned.
    """

    @classmethod
    @abstractmethod
    def get_parameter(cls) -> List[Dict]:
        """
        The parameters list to initialize encoder networks.
        """
        pass

    @classmethod
    @abstractmethod
    def get_output_feature_channel_list(cls) -> List[Tuple[int, ...]]:
        """
        Get number of output feature channels.
        """
        pass

    @classmethod
    @abstractmethod
    def get_output_feature_number_list(cls) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def get_encoder_name_string_list(cls) -> List[str]:
        pass
