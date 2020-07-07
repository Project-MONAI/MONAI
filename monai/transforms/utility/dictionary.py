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

from logging import Handler
from typing import Optional, Callable, Union, Sequence
import copy
import torch
import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.transforms.utility.array import (
    AddChannel,
    AsChannelFirst,
    ToTensor,
    ToNumpy,
    AsChannelLast,
    CastToType,
    RepeatChannel,
    SqueezeDim,
    DataStats,
    SimulateDelay,
    Identity,
    Lambda,
)


class Identityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    def __init__(self, keys: KeysCollection):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`

        """
        super().__init__(keys)
        self.identity = Identity()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.identity(d[key])
        return d


class AsChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelFirst`.
    """

    def __init__(self, keys: KeysCollection, channel_dim: int = -1):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim: which dimension of input image is the channel, default is the last dimension.
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

    def __init__(self, keys: KeysCollection, channel_dim: int = 0):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim: which dimension of input image is the channel, default is the first dimension.
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

    def __init__(self, keys: KeysCollection):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
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
    Dictionary-based wrapper of :py:class:`monai.transforms.RepeatChannel`.
    """

    def __init__(self, keys: KeysCollection, repeats: int):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            repeats: the number of repetitions for each element.
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

    def __init__(self, keys: KeysCollection, dtype: Union[Sequence[np.dtype], np.dtype] = np.float32):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of np.dtype, each element corresponds to a key in ``keys``.

        """
        MapTransform.__init__(self, keys)
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.converter = CastToType()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key], dtype=self.dtype[idx])

        return d


class ToTensord(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    def __init__(self, keys: KeysCollection):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.converter = ToTensor()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class ToNumpyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToNumpy`.
    """

    def __init__(self, keys: KeysCollection):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.converter = ToNumpy()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class DeleteItemsd(MapTransform):
    """
    Delete specified items from data dictionary to release memory.
    It will remove the key-values and copy the others to construct a new dictionary.
    """

    def __init__(self, keys: KeysCollection):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)

    def __call__(self, data):
        return {key: val for key, val in data.items() if key not in self.keys}


class SqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SqueezeDim`.
    """

    def __init__(self, keys: KeysCollection, dim: int = 0):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dim: dimension to be squeezed. Default: 0 (the first dimension)
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
        keys: KeysCollection,
        prefix: Union[Sequence[str], str] = "Data",
        data_shape: Union[Sequence[bool], bool] = True,
        value_range: Union[Sequence[bool], bool] = True,
        data_value: Union[Sequence[bool], bool] = False,
        additional_info: Optional[Union[Sequence[Callable], Callable]] = None,
        logger_handler: Optional[Handler] = None,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prefix: will be printed in format: "{prefix} statistics".
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
            data_shape: whether to show the shape of input data.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            value_range: whether to show the value range of input data.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            data_value: whether to show the raw value of input data.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
                a typical example is to print some properties of Nifti image: affine, pixdim, etc.
            additional_info: user can define callable function to extract
                additional info from input data. it also can be a sequence of string, each element
                corresponds to a key in ``keys``.
            logger_handler: add additional handler to output data: save to file, etc.
                add existing python logging handlers: https://docs.python.org/3/library/logging.handlers.html

        """
        super().__init__(keys)
        self.prefix = ensure_tuple_rep(prefix, len(self.keys))
        self.data_shape = ensure_tuple_rep(data_shape, len(self.keys))
        self.value_range = ensure_tuple_rep(value_range, len(self.keys))
        self.data_value = ensure_tuple_rep(data_value, len(self.keys))
        self.additional_info = ensure_tuple_rep(additional_info, len(self.keys))
        self.logger_handler = logger_handler
        self.printer = DataStats(logger_handler=logger_handler)

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.printer(
                d[key],
                self.prefix[idx],
                self.data_shape[idx],
                self.value_range[idx],
                self.data_value[idx],
                self.additional_info[idx],
            )
        return d


class SimulateDelayd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:monai.transforms.utility.array.SimulateDelay.
    """

    def __init__(self, keys: KeysCollection, delay_time: Union[Sequence[float], float] = 0.0):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            delay_time: The minimum amount of time, in fractions of seconds, to accomplish this identity task.
                It also can be a sequence of string, each element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.delay_time = ensure_tuple_rep(delay_time, len(self.keys))
        self.delayer = SimulateDelay()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.delayer(d[key], delay_time=self.delay_time[idx])
        return d


class CopyItemsd(MapTransform):
    """
    Copy specified items from data dictionary and save with different key names.
    It can copy several items together and copy several times.

    """

    def __init__(self, keys: KeysCollection, times: int, names: KeysCollection):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            times: expected copy times, for example, if keys is "img", times is 3,
                it will add 3 copies of "img" data to the dictionary.
            names: the names coresponding to the newly copied data,
                the length should match `len(keys) x times`. for example, if keys is ["img", "seg"]
                and times is 2, names can be: ["img_1", "seg_1", "img_2", "seg_2"].

        Raises:
            ValueError: times must be greater than 0.
            ValueError: length of names does not match `len(keys) x times`.

        """
        super().__init__(keys)
        if times < 1:
            raise ValueError("times must be greater than 0.")
        self.times = times
        names = ensure_tuple(names)
        if len(names) != (len(self.keys) * times):
            raise ValueError("length of names does not match `len(keys) x times`.")
        self.names = names

    def __call__(self, data):
        d = dict(data)
        for key, new_key in zip(self.keys * self.times, self.names):
            if new_key in d:
                raise KeyError(f"key {new_key} already exists in dictionary.")
            d[new_key] = copy.deepcopy(d[key])
        return d


class ConcatItemsd(MapTransform):
    """
    Concatenate specified items from data dictionary together on the first dim to construct a big array.
    Expect all the items are numpy array or PyTorch Tensor.

    """

    def __init__(self, keys: KeysCollection, name: str, dim: int = 0):
        """
        Args:
            keys: keys of the corresponding items to be concatenated together.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name coresponding to the key to store the concatenated data.
            dim: on which dimension to concatenate the items, default is 0.

        Raises:
            ValueError: must provide must than 1 items to concat.

        """
        super().__init__(keys)
        if len(self.keys) < 2:
            raise ValueError("must provide must than 1 items to concat.")
        self.name = name
        self.dim = dim

    def __call__(self, data):
        d = dict(data)
        output = list()
        data_type = None
        for key in self.keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("not all the items are with same data type.")
            output.append(d[key])
        if data_type == np.ndarray:
            d[self.name] = np.concatenate(output, axis=self.dim)
        elif data_type == torch.Tensor:
            d[self.name] = torch.cat(output, dim=self.dim)
        else:
            raise TypeError(f"unsupported data type to concat: {data_type}.")
        return d


class Lambdad(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Lambda`.

    For example:

    .. code-block:: python
        :emphasize-lines: 2

        input_data={'image': np.zeros((10, 2, 2)), 'label': np.ones((10, 2, 2))}
        lambd = Lambdad(keys='label', func=lambda x: x[:4, :, :])
        print(lambd(input_data)['label'].shape)
        (4, 2, 2)

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        func: Lambda/function to be applied. It also can be a sequence of Callable,
            each element corresponds to a key in ``keys``.
    """

    def __init__(self, keys: KeysCollection, func: Callable) -> None:
        super().__init__(keys)
        self.func = ensure_tuple_rep(func, len(self.keys))
        self.lambd = Lambda()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.lambd(d[key], func=self.func[idx])

        return d


IdentityD = IdentityDict = Identityd
AsChannelFirstD = AsChannelFirstDict = AsChannelFirstd
AsChannelLastD = AsChannelLastDict = AsChannelLastd
AddChannelD = AddChannelDict = AddChanneld
RepeatChannelD = RepeatChannelDict = RepeatChanneld
CastToTypeD = CastToTypeDict = CastToTyped
ToTensorD = ToTensorDict = ToTensord
DeleteItemsD = DeleteItemsDict = DeleteItemsd
SqueezeDimD = SqueezeDimDict = SqueezeDimd
DataStatsD = DataStatsDict = DataStatsd
SimulateDelayD = SimulateDelayDict = SimulateDelayd
CopyItemsD = CopyItemsDict = CopyItemsd
ConcatItemsD = ConcatItemsDict = ConcatItemsd
LambdaD = LambdaDict = Lambdad
