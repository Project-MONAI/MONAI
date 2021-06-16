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
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for utility functions
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import copy
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection, NdarrayTensor
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utility.array import (
    AddChannel,
    AsChannelFirst,
    AsChannelLast,
    CastToType,
    ConvertToMultiChannelBasedOnBratsClasses,
    DataStats,
    EnsureChannelFirst,
    FgBgToIndices,
    Identity,
    LabelToMask,
    Lambda,
    MapLabelValue,
    RemoveRepeatedChannel,
    RepeatChannel,
    SimulateDelay,
    SplitChannel,
    SqueezeDim,
    ToCupy,
    ToNumpy,
    ToPIL,
    TorchVision,
    ToTensor,
    Transpose,
)
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils.enums import InverseKeys

__all__ = [
    "AddChannelD",
    "AddChannelDict",
    "AddChanneld",
    "AddExtremePointsChannelD",
    "AddExtremePointsChannelDict",
    "AddExtremePointsChanneld",
    "AsChannelFirstD",
    "AsChannelFirstDict",
    "AsChannelFirstd",
    "AsChannelLastD",
    "AsChannelLastDict",
    "AsChannelLastd",
    "CastToTypeD",
    "CastToTypeDict",
    "CastToTyped",
    "ConcatItemsD",
    "ConcatItemsDict",
    "ConcatItemsd",
    "ConvertToMultiChannelBasedOnBratsClassesD",
    "ConvertToMultiChannelBasedOnBratsClassesDict",
    "ConvertToMultiChannelBasedOnBratsClassesd",
    "CopyItemsD",
    "CopyItemsDict",
    "CopyItemsd",
    "DataStatsD",
    "DataStatsDict",
    "DataStatsd",
    "DeleteItemsD",
    "DeleteItemsDict",
    "DeleteItemsd",
    "EnsureChannelFirstD",
    "EnsureChannelFirstDict",
    "EnsureChannelFirstd",
    "FgBgToIndicesD",
    "FgBgToIndicesDict",
    "FgBgToIndicesd",
    "IdentityD",
    "IdentityDict",
    "Identityd",
    "LabelToMaskD",
    "LabelToMaskDict",
    "LabelToMaskd",
    "LambdaD",
    "LambdaDict",
    "Lambdad",
    "MapLabelValueD",
    "MapLabelValueDict",
    "MapLabelValued",
    "RandLambdaD",
    "RandLambdaDict",
    "RandLambdad",
    "RandTorchVisionD",
    "RandTorchVisionDict",
    "RandTorchVisiond",
    "RemoveRepeatedChannelD",
    "RemoveRepeatedChannelDict",
    "RemoveRepeatedChanneld",
    "RepeatChannelD",
    "RepeatChannelDict",
    "RepeatChanneld",
    "SelectItemsD",
    "SelectItemsDict",
    "SelectItemsd",
    "SimulateDelayD",
    "SimulateDelayDict",
    "SimulateDelayd",
    "SplitChannelD",
    "SplitChannelDict",
    "SplitChanneld",
    "SqueezeDimD",
    "SqueezeDimDict",
    "SqueezeDimd",
    "ToCupyD",
    "ToCupyDict",
    "ToCupyd",
    "ToNumpyD",
    "ToNumpyDict",
    "ToNumpyd",
    "ToPILD",
    "ToPILDict",
    "ToPILd",
    "ToTensorD",
    "ToTensorDict",
    "ToTensord",
    "TorchVisionD",
    "TorchVisionDict",
    "TorchVisiond",
    "Transposed",
    "TransposeDict",
    "TransposeD",
]


class Identityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.identity = Identity()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.identity(d[key])
        return d


class AsChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelFirst`.
    """

    def __init__(self, keys: KeysCollection, channel_dim: int = -1, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim: which dimension of input image is the channel, default is the last dimension.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = AsChannelFirst(channel_dim=channel_dim)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class AsChannelLastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelLast`.
    """

    def __init__(self, keys: KeysCollection, channel_dim: int = 0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim: which dimension of input image is the channel, default is the first dimension.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = AsChannelLast(channel_dim=channel_dim)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.adder = AddChannel()

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d


class EnsureChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.EnsureChannelFirst`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        strict_check: bool = True,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None and `key_{postfix}` was used to store the metadata in `LoadImaged`.
                So need the key to extract metadata for channel dim information, default is `meta_dict`.
                For example, for data with key `image`, metadata by default is in `image_meta_dict`.
            strict_check: whether to raise an error when the meta information is insufficient.

        """
        super().__init__(keys)
        self.adjuster = EnsureChannelFirst(strict_check=strict_check)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, meta_key, meta_key_postfix in zip(self.keys, self.meta_keys, self.meta_key_postfix):
            d[key] = self.adjuster(d[key], d[meta_key or f"{key}_{meta_key_postfix}"])
        return d


class RepeatChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RepeatChannel`.
    """

    def __init__(self, keys: KeysCollection, repeats: int, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            repeats: the number of repetitions for each element.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.repeater = RepeatChannel(repeats)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.repeater(d[key])
        return d


class RemoveRepeatedChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RemoveRepeatedChannel`.
    """

    def __init__(self, keys: KeysCollection, repeats: int, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            repeats: the number of repetitions for each element.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.repeater = RemoveRepeatedChannel(repeats)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.repeater(d[key])
        return d


class SplitChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SplitChannel`.
    All the input specified by `keys` should be split into same count of data.

    """

    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Optional[Sequence[str]] = None,
        channel_dim: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            output_postfixes: the postfixes to construct keys to store split data.
                for example: if the key of input data is `pred` and split 2 classes, the output
                data keys will be: pred_(output_postfixes[0]), pred_(output_postfixes[1])
                if None, using the index number: `pred_0`, `pred_1`, ... `pred_N`.
            channel_dim: which dimension of input image is the channel, default to None
                to automatically select: if data is numpy array, channel_dim is 0 as
                `numpy array` is used in the pre transforms, if PyTorch Tensor, channel_dim
                is 1 as in most of the cases `Tensor` is uses in the post transforms.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.output_postfixes = output_postfixes
        self.splitter = SplitChannel(channel_dim=channel_dim)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            rets = self.splitter(d[key])
            postfixes: Sequence = list(range(len(rets))) if self.output_postfixes is None else self.output_postfixes
            if len(postfixes) != len(rets):
                raise AssertionError("count of split results must match output_postfixes.")
            for i, r in enumerate(rets):
                split_key = f"{key}_{postfixes[i]}"
                if split_key in d:
                    raise RuntimeError(f"input data already contains key {split_key}.")
                d[split_key] = r
        return d


class CastToTyped(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CastToType`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Union[Sequence[Union[DtypeLike, torch.dtype]], DtypeLike, torch.dtype] = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of dtypes or torch.dtype,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.converter = CastToType()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key, dtype in self.key_iterator(d, self.dtype):
            d[key] = self.converter(d[key], dtype=dtype)

        return d


class ToTensord(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToTensor()

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # Create inverse transform
            inverse_transform = ToNumpy()
            # Apply inverse
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class ToNumpyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToNumpy`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToNumpy()

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ToCupyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToCupy`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToCupy()

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ToPILd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToNumpy`.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToPIL()

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class Transposed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Transpose`.
    """

    def __init__(
        self, keys: KeysCollection, indices: Optional[Sequence[int]], allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = Transpose(indices)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
            # if None was supplied then numpy uses range(a.ndim)[::-1]
            indices = self.transform.indices or range(d[key].ndim)[::-1]
            self.push_transform(d, key, extra_info={"indices": indices})
        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            fwd_indices = np.array(transform[InverseKeys.EXTRA_INFO]["indices"])
            inv_indices = np.argsort(fwd_indices)
            inverse_transform = Transpose(inv_indices.tolist())
            # Apply inverse
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class DeleteItemsd(MapTransform):
    """
    Delete specified items from data dictionary to release memory.
    It will remove the key-values and copy the others to construct a new dictionary.
    """

    def __call__(self, data):
        return {key: val for key, val in data.items() if key not in self.key_iterator(data)}


class SelectItemsd(MapTransform):
    """
    Select only specified items from data dictionary to release memory.
    It will copy the selected key-values and construct and new dictionary.
    """

    def __call__(self, data):
        result = {key: data[key] for key in self.key_iterator(data)}
        return result


class SqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SqueezeDim`.
    """

    def __init__(self, keys: KeysCollection, dim: int = 0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dim: dimension to be squeezed. Default: 0 (the first dimension)
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = SqueezeDim(dim=dim)

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
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
        data_type: Union[Sequence[bool], bool] = True,
        data_shape: Union[Sequence[bool], bool] = True,
        value_range: Union[Sequence[bool], bool] = True,
        data_value: Union[Sequence[bool], bool] = False,
        additional_info: Optional[Union[Sequence[Callable], Callable]] = None,
        logger_handler: Optional[logging.Handler] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prefix: will be printed in format: "{prefix} statistics".
                it also can be a sequence of string, each element corresponds to a key in ``keys``.
            data_type: whether to show the type of input data.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
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
                the handler should have a logging level of at least `INFO`.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.prefix = ensure_tuple_rep(prefix, len(self.keys))
        self.data_type = ensure_tuple_rep(data_type, len(self.keys))
        self.data_shape = ensure_tuple_rep(data_shape, len(self.keys))
        self.value_range = ensure_tuple_rep(value_range, len(self.keys))
        self.data_value = ensure_tuple_rep(data_value, len(self.keys))
        self.additional_info = ensure_tuple_rep(additional_info, len(self.keys))
        self.logger_handler = logger_handler
        self.printer = DataStats(logger_handler=logger_handler)

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key, prefix, data_type, data_shape, value_range, data_value, additional_info in self.key_iterator(
            d, self.prefix, self.data_type, self.data_shape, self.value_range, self.data_value, self.additional_info
        ):
            d[key] = self.printer(
                d[key],
                prefix,
                data_type,
                data_shape,
                value_range,
                data_value,
                additional_info,
            )
        return d


class SimulateDelayd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SimulateDelay`.
    """

    def __init__(
        self, keys: KeysCollection, delay_time: Union[Sequence[float], float] = 0.0, allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            delay_time: The minimum amount of time, in fractions of seconds, to accomplish this identity task.
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.delay_time = ensure_tuple_rep(delay_time, len(self.keys))
        self.delayer = SimulateDelay()

    def __call__(self, data: Mapping[Hashable, NdarrayTensor]) -> Dict[Hashable, NdarrayTensor]:
        d = dict(data)
        for key, delay_time in self.key_iterator(d, self.delay_time):
            d[key] = self.delayer(d[key], delay_time=delay_time)
        return d


class CopyItemsd(MapTransform):
    """
    Copy specified items from data dictionary and save with different key names.
    It can copy several items together and copy several times.

    """

    def __init__(
        self, keys: KeysCollection, times: int, names: KeysCollection, allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            times: expected copy times, for example, if keys is "img", times is 3,
                it will add 3 copies of "img" data to the dictionary.
            names: the names corresponding to the newly copied data,
                the length should match `len(keys) x times`. for example, if keys is ["img", "seg"]
                and times is 2, names can be: ["img_1", "seg_1", "img_2", "seg_2"].
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            ValueError: When ``times`` is nonpositive.
            ValueError: When ``len(names)`` is not ``len(keys) * times``. Incompatible values.

        """
        super().__init__(keys, allow_missing_keys)
        if times < 1:
            raise ValueError(f"times must be positive, got {times}.")
        self.times = times
        names = ensure_tuple(names)
        if len(names) != (len(self.keys) * times):
            raise ValueError(
                "len(names) must match len(keys) * times, "
                f"got len(names)={len(names)} len(keys) * times={len(self.keys) * times}."
            )
        self.names = names

    def __call__(self, data):
        """
        Raises:
            KeyError: When a key in ``self.names`` already exists in ``data``.

        """
        d = dict(data)
        key_len = len(self.keys)
        for i in range(self.times):
            for key, new_key in self.key_iterator(d, self.names[i * key_len : (i + 1) * key_len]):
                if new_key in d:
                    raise KeyError(f"Key {new_key} already exists in data.")
                if isinstance(d[key], torch.Tensor):
                    d[new_key] = d[key].detach().clone()
                else:
                    d[new_key] = copy.deepcopy(d[key])
        return d


class ConcatItemsd(MapTransform):
    """
    Concatenate specified items from data dictionary together on the first dim to construct a big array.
    Expect all the items are numpy array or PyTorch Tensor.

    """

    def __init__(self, keys: KeysCollection, name: str, dim: int = 0, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be concatenated together.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name corresponding to the key to store the concatenated data.
            dim: on which dimension to concatenate the items, default is 0.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.name = name
        self.dim = dim

    def __call__(self, data):
        """
        Raises:
            TypeError: When items in ``data`` differ in type.
            TypeError: When the item type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        d = dict(data)
        output = []
        data_type = None
        for key in self.key_iterator(d):
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])
        if data_type == np.ndarray:
            d[self.name] = np.concatenate(output, axis=self.dim)
        elif data_type == torch.Tensor:
            d[self.name] = torch.cat(output, dim=self.dim)
        else:
            raise TypeError(f"Unsupported data type: {data_type}, available options are (numpy.ndarray, torch.Tensor).")
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
        overwrite: whether to overwrite the original data in the input dictionary with lamdbda function output.
            default to True. it also can be a sequence of bool, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        func: Union[Sequence[Callable], Callable],
        overwrite: Union[Sequence[bool], bool] = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.func = ensure_tuple_rep(func, len(self.keys))
        self.overwrite = ensure_tuple_rep(overwrite, len(self.keys))
        self._lambd = Lambda()

    def __call__(self, data):
        d = dict(data)
        for key, func, overwrite in self.key_iterator(d, self.func, self.overwrite):
            ret = self._lambd(d[key], func=func)
            if overwrite:
                d[key] = ret
        return d


class RandLambdad(Lambdad, Randomizable):
    """
    Randomizable version :py:class:`monai.transforms.Lambdad`, the input `func` contains random logic.
    It's a randomizable transform so `CacheDataset` will not execute it and cache the results.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        func: Lambda/function to be applied. It also can be a sequence of Callable,
            each element corresponds to a key in ``keys``.
        overwrite: whether to overwrite the original data in the input dictionary with lamdbda function output.
            default to True. it also can be a sequence of bool, each element corresponds to a key in ``keys``.

    For more details, please check :py:class:`monai.transforms.Lambdad`.

    """

    def randomize(self, data: Any) -> None:
        pass


class LabelToMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LabelToMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
            is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
            `select_labels` is the expected channel indices.
        merge_channels: whether to use `np.any()` to merge the result on channel dim.
            if yes, will return a single channel mask with binary data.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        keys: KeysCollection,
        select_labels: Union[Sequence[int], int],
        merge_channels: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:  # pytype: disable=annotation-type-mismatch
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelToMask(select_labels=select_labels, merge_channels=merge_channels)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])

        return d


class FgBgToIndicesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.FgBgToIndices`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        fg_postfix: postfix to save the computed foreground indices in dict.
            for example, if computed on `label` and `postfix = "_fg_indices"`, the key will be `label_fg_indices`.
        bg_postfix: postfix to save the computed background indices in dict.
            for example, if computed on `label` and `postfix = "_bg_indices"`, the key will be `label_bg_indices`.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to determine
            the negative sample(background). so the output items will not map to all the voxels in the label.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area and select background only in this area.
        output_shape: expected shape of output indices. if not None, unravel indices to specified shape.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        fg_postfix: str = "_fg_indices",
        bg_postfix: str = "_bg_indices",
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        output_shape: Optional[Sequence[int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.fg_postfix = fg_postfix
        self.bg_postfix = bg_postfix
        self.image_key = image_key
        self.converter = FgBgToIndices(image_threshold, output_shape)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        image = d[self.image_key] if self.image_key else None
        for key in self.key_iterator(d):
            d[str(key) + self.fg_postfix], d[str(key) + self.bg_postfix] = self.converter(d[key], image)

        return d


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses()

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class AddExtremePointsChanneld(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddExtremePointsChannel`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: key to label source to get the extreme points.
        background: Class index of background label, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.
        sigma: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        rescale_min: minimum value of output data.
        rescale_max: maximum value of output data.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        background: int = 0,
        pert: float = 0.0,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 3.0,
        rescale_min: float = -1.0,
        rescale_max: float = 1.0,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.background = background
        self.pert = pert
        self.points: List[Tuple[int, ...]] = []
        self.label_key = label_key
        self.sigma = sigma
        self.rescale_min = rescale_min
        self.rescale_max = rescale_max

    def randomize(self, label: np.ndarray) -> None:
        self.points = get_extreme_points(label, rand_state=self.R, background=self.background, pert=self.pert)

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        if label.shape[0] != 1:
            raise ValueError("Only supports single channel labels!")

        # Generate extreme points
        self.randomize(label[0, :])

        for key in self.key_iterator(d):
            img = d[key]
            points_image = extreme_points_to_image(
                points=self.points,
                label=label,
                sigma=self.sigma,
                rescale_min=self.rescale_min,
                rescale_max=self.rescale_max,
            )
            d[key] = np.concatenate([img, points_image], axis=0)
        return d


class TorchVisiond(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.TorchVision` for non-randomized transforms.
    For randomized transforms of TorchVision use :py:class:`monai.transforms.RandTorchVisiond`.

    Note:
        As most of the TorchVision transforms only work for PIL image and PyTorch Tensor, this transform expects input
        data to be dict of PyTorch Tensors, users can easily call `ToTensord` transform to convert Numpy to Tensor.
    """

    def __init__(
        self,
        keys: KeysCollection,
        name: str,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: The transform name in TorchVision package.
            allow_missing_keys: don't raise exception if key is missing.
            args: parameters for the TorchVision transform.
            kwargs: parameters for the TorchVision transform.

        """
        super().__init__(keys, allow_missing_keys)
        self.trans = TorchVision(name, *args, **kwargs)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.trans(d[key])
        return d


class RandTorchVisiond(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.TorchVision` for randomized transforms.
    For deterministic non-randomized transforms of TorchVision use :py:class:`monai.transforms.TorchVisiond`.

    Note:

        - As most of the TorchVision transforms only work for PIL image and PyTorch Tensor, this transform expects input
          data to be dict of PyTorch Tensors, users can easily call `ToTensord` transform to convert Numpy to Tensor.
        - This class inherits the ``Randomizable`` purely to prevent any dataset caching to skip the transform
          computation. If the random factor of the underlying torchvision transform is not derived from `self.R`,
          the results may not be deterministic.
          See Also: :py:class:`monai.transforms.Randomizable`.

    """

    def __init__(
        self,
        keys: KeysCollection,
        name: str,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: The transform name in TorchVision package.
            allow_missing_keys: don't raise exception if key is missing.
            args: parameters for the TorchVision transform.
            kwargs: parameters for the TorchVision transform.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.trans = TorchVision(name, *args, **kwargs)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.trans(d[key])
        return d


class MapLabelValued(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MapLabelValue`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        orig_labels: Sequence,
        target_labels: Sequence,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            orig_labels: original labels that map to others.
            target_labels: expected label values, 1: 1 map to the `orig_labels`.
            dtype: convert the output data to dtype, default to float32.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.mapper = MapLabelValue(orig_labels=orig_labels, target_labels=target_labels, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.mapper(d[key])
        return d


IdentityD = IdentityDict = Identityd
AsChannelFirstD = AsChannelFirstDict = AsChannelFirstd
AsChannelLastD = AsChannelLastDict = AsChannelLastd
AddChannelD = AddChannelDict = AddChanneld
EnsureChannelFirstD = EnsureChannelFirstDict = EnsureChannelFirstd
RemoveRepeatedChannelD = RemoveRepeatedChannelDict = RemoveRepeatedChanneld
RepeatChannelD = RepeatChannelDict = RepeatChanneld
SplitChannelD = SplitChannelDict = SplitChanneld
CastToTypeD = CastToTypeDict = CastToTyped
ToTensorD = ToTensorDict = ToTensord
ToNumpyD = ToNumpyDict = ToNumpyd
ToCupyD = ToCupyDict = ToCupyd
ToPILD = ToPILDict = ToPILd
TransposeD = TransposeDict = Transposed
DeleteItemsD = DeleteItemsDict = DeleteItemsd
SelectItemsD = SelectItemsDict = SelectItemsd
SqueezeDimD = SqueezeDimDict = SqueezeDimd
DataStatsD = DataStatsDict = DataStatsd
SimulateDelayD = SimulateDelayDict = SimulateDelayd
CopyItemsD = CopyItemsDict = CopyItemsd
ConcatItemsD = ConcatItemsDict = ConcatItemsd
LambdaD = LambdaDict = Lambdad
LabelToMaskD = LabelToMaskDict = LabelToMaskd
FgBgToIndicesD = FgBgToIndicesDict = FgBgToIndicesd
ConvertToMultiChannelBasedOnBratsClassesD = (
    ConvertToMultiChannelBasedOnBratsClassesDict
) = ConvertToMultiChannelBasedOnBratsClassesd
AddExtremePointsChannelD = AddExtremePointsChannelDict = AddExtremePointsChanneld
TorchVisionD = TorchVisionDict = TorchVisiond
RandTorchVisionD = RandTorchVisionDict = RandTorchVisiond
RandLambdaD = RandLambdaDict = RandLambdad
MapLabelValueD = MapLabelValueDict = MapLabelValued
