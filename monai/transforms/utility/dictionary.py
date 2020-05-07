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
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for utility functions
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import numpy as np

from monai.transforms.compose import MapTransform
from monai.transforms.utility.array import (
    AddChannel,
    AsChannelFirst,
    ToTensor,
    AsChannelLast,
    CastToType,
    RepeatChannel,
    SqueezeDim,
    DataStats,
    SimulateDelay,
)


class AsChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelFirst`.
    """

    def __init__(self, keys, channel_dim=-1):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim (int): which dimension of input image is the channel, default is the last dimension.
        """
        super().__init__(keys)
        self.converter = AsChannelFirst(channel_dim=channel_dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AsChannelLastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelLast`.
    """

    def __init__(self, keys, channel_dim=0):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim (int): which dimension of input image is the channel, default is the first dimension.
        """
        super().__init__(keys)
        self.converter = AsChannelLast(channel_dim=channel_dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.adder = AddChannel()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.adder(d[key])
        return d


class RepeatChanneld(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.RepeatChannel`.
    """

    def __init__(self, keys, repeats):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            repeats (int): the number of repetitions for each element.
        """
        super().__init__(keys)
        self.repeater = RepeatChannel(repeats)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.repeater(d[key])
        return d


class CastToTyped(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CastToType`.
    """

    def __init__(self, keys, dtype=np.float32):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype (np.dtype): convert image to this data type, default is `np.float32`.
        """
        MapTransform.__init__(self, keys)
        self.converter = CastToType(dtype)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class ToTensord(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.converter = ToTensor()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class DeleteKeysd(MapTransform):
    """
    Delete specified keys from data dictionary to release memory.
    It will remove the key-values and copy the others to construct a new dictionary.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)

    def __call__(self, data):
        return {key: val for key, val in data.items() if key not in self.keys}


class SqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SqueezeDim`.
    """

    def __init__(self, keys, dim=None):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dim (int): dimension to be squeezed.
                Default: None (all dimensions of size 1 will be removed)
        """
        super().__init__(keys)
        self.converter = SqueezeDim(dim=dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class DataStatsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.DataStats`.
    """

    def __init__(
        self,
        keys,
        prefix="Data",
        data_shape=True,
        intensity_range=True,
        data_value=False,
        additional_info=None,
        logger_handler=None,
    ):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prefix (string): will be printed in format: "{prefix} statistics".
            data_shape (bool): whether to show the shape of input data.
            intensity_range (bool): whether to show the intensity value range of input data.
            data_value (bool): whether to show the raw value of input data.
                a typical example is to print some properties of Nifti image: affine, pixdim, etc.
            additional_info (Callable): user can define callable function to extract additional info from input data.
            logger_handler (logging.handler): add additional handler to output data: save to file, etc.
                add existing python logging handlers: https://docs.python.org/3/library/logging.handlers.html
        """
        super().__init__(keys)
        self.printer = DataStats(prefix, data_shape, intensity_range, data_value, additional_info, logger_handler)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.printer(d[key])
        return d


class SimulateDelayd(MapTransform):
    """
    dictionary-based wrapper of :py:class:monai.transforms.utility.array.SimulateDelay.
    """

    def __init__(self, keys, delay_time=0.0):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            delay_time(float or list of float): The minimum amount of time, in fractions of seconds,
                to accomplish this identity task. If a list is provided, it must be of length equal
                to the keys representing the delay for each key element.
        """
        super().__init__(keys)

        if not isinstance(delay_time, (tuple, list)):
            delay_time = [delay_time] * len(keys)
        self.converter_dictionary = dict()
        for key, keyed_default_delay in zip(self.keys, delay_time):
            self.converter_dictionary[key] = SimulateDelay(delay_time=keyed_default_delay)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter_dictionary[key](d[key])
        return d


AsChannelFirstD = AsChannelFirstDict = AsChannelFirstd
AsChannelLastD = AsChannelLastDict = AsChannelLastd
AddChannelD = AddChannelDict = AddChanneld
RepeatChannelD = RepeatChannelDict = RepeatChanneld
CastToTypeD = CastToTypeDict = CastToTyped
ToTensorD = ToTensorDict = ToTensord
DeleteKeysD = DeleteKeysDict = DeleteKeysd
SqueezeDimD = SqueezeDimDict = SqueezeDimd
DataStatsD = DataStatsDict = DataStatsd
SimulateDelayD = SimulateDelayDict = SimulateDelayd
