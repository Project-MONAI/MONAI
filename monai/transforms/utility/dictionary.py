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
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for utility functions
defined in :py:class:`monai.transforms.utility.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import re
from copy import deepcopy
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import no_collation
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable, RandomizableTransform
from monai.transforms.utility.array import (
    AddChannel,
    AddCoordinateChannels,
    AddExtremePointsChannel,
    AsChannelFirst,
    AsChannelLast,
    CastToType,
    ClassesToIndices,
    ConvertToMultiChannelBasedOnBratsClasses,
    CuCIM,
    DataStats,
    EnsureChannelFirst,
    EnsureType,
    FgBgToIndices,
    Identity,
    IntensityStats,
    LabelToMask,
    Lambda,
    MapLabelValue,
    RemoveRepeatedChannel,
    RepeatChannel,
    SimulateDelay,
    SplitDim,
    SqueezeDim,
    ToCupy,
    ToDevice,
    ToNumpy,
    ToPIL,
    TorchVision,
    ToTensor,
    Transpose,
)
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.transforms.utils_pytorch_numpy_unification import concatenate
from monai.utils import convert_to_numpy, deprecated, deprecated_arg, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix, TraceKeys, TransformBackends
from monai.utils.type_conversion import convert_to_dst_type

__all__ = [
    "AddChannelD",
    "AddChannelDict",
    "AddChanneld",
    "AddCoordinateChannelsD",
    "AddCoordinateChannelsDict",
    "AddCoordinateChannelsd",
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
    "CuCIMd",
    "CuCIMD",
    "CuCIMDict",
    "DataStatsD",
    "DataStatsDict",
    "DataStatsd",
    "DeleteItemsD",
    "DeleteItemsDict",
    "DeleteItemsd",
    "EnsureChannelFirstD",
    "EnsureChannelFirstDict",
    "EnsureChannelFirstd",
    "EnsureTypeD",
    "EnsureTypeDict",
    "EnsureTyped",
    "FgBgToIndicesD",
    "FgBgToIndicesDict",
    "FgBgToIndicesd",
    "IdentityD",
    "IdentityDict",
    "Identityd",
    "IntensityStatsd",
    "IntensityStatsD",
    "IntensityStatsDict",
    "LabelToMaskD",
    "LabelToMaskDict",
    "LabelToMaskd",
    "LambdaD",
    "LambdaDict",
    "Lambdad",
    "MapLabelValueD",
    "MapLabelValueDict",
    "MapLabelValued",
    "RandCuCIMd",
    "RandCuCIMD",
    "RandCuCIMDict",
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
    "SplitDimD",
    "SplitDimDict",
    "SplitDimd",
    "SqueezeDimD",
    "SqueezeDimDict",
    "SqueezeDimd",
    "ToCupyD",
    "ToCupyDict",
    "ToCupyd",
    "ToDeviced",
    "ToDeviceD",
    "ToDeviceDict",
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
    "ClassesToIndicesd",
    "ClassesToIndicesD",
    "ClassesToIndicesDict",
]

DEFAULT_POST_FIX = PostFix.meta()


class Identityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    backend = Identity.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.identity = Identity()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.identity(d[key])
        return d


class AsChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelFirst`.
    """

    backend = AsChannelFirst.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class AsChannelLastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelLast`.
    """

    backend = AsChannelLast.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = AddChannel.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.adder = AddChannel()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d


class EnsureChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.EnsureChannelFirst`.
    """

    backend = EnsureChannelFirst.backend

    @deprecated_arg(name="meta_keys", since="0.8", msg_suffix="not needed if image is type `MetaTensor`.")
    @deprecated_arg(name="meta_key_postfix", since="0.8", msg_suffix="not needed if image is type `MetaTensor`.")
    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        strict_check: bool = True,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            strict_check: whether to raise an error when the meta information is insufficient.

        """
        super().__init__(keys)
        self.adjuster = EnsureChannelFirst(strict_check=strict_check)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adjuster(d[key])
        return d


class RepeatChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RepeatChannel`.
    """

    backend = RepeatChannel.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.repeater(d[key])
        return d


class RemoveRepeatedChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RemoveRepeatedChannel`.
    """

    backend = RemoveRepeatedChannel.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.repeater(d[key])
        return d


class SplitDimd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Optional[Sequence[str]] = None,
        dim: int = 0,
        keepdim: bool = True,
        update_meta: bool = True,
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
            dim: which dimension of input image is the channel, default to 0.
            keepdim: if `True`, output will have singleton in the split dimension. If `False`, this
                dimension will be squeezed.
            update_meta: if `True`, copy `[key]_meta_dict` for each output and update affine to
                reflect the cropped image
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.output_postfixes = output_postfixes
        self.splitter = SplitDim(dim, keepdim, update_meta)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            rets = self.splitter(d[key])
            postfixes: Sequence = list(range(len(rets))) if self.output_postfixes is None else self.output_postfixes
            if len(postfixes) != len(rets):
                raise ValueError(f"count of splits must match output_postfixes, {len(postfixes)} != {len(rets)}.")
            for i, r in enumerate(rets):
                split_key = f"{key}_{postfixes[i]}"
                if split_key in d:
                    raise RuntimeError(f"input data already contains key {split_key}.")
                d[split_key] = r
        return d


@deprecated(since="0.8", msg_suffix="please use `SplitDimd` instead.")
class SplitChanneld(SplitDimd):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SplitChannel`.
    All the input specified by `keys` should be split into same count of data.
    """

    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Optional[Sequence[str]] = None,
        channel_dim: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(
            keys,
            output_postfixes=output_postfixes,
            dim=channel_dim,
            update_meta=False,  # for backwards compatibility
            allow_missing_keys=allow_missing_keys,
        )


class CastToTyped(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CastToType`.
    """

    backend = CastToType.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Union[Sequence[Union[DtypeLike, torch.dtype]], DtypeLike, torch.dtype] = np.float32,
        drop_meta: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of dtypes or torch.dtype,
                each element corresponds to a key in ``keys``.
            drop_meta: whether to drop the meta information of the input data, default to `True`.
                If `True`, then the meta information will be dropped quietly, unless the output type is MetaTensor.
                If `False`, converting a MetaTensor into a non-tensor instance will raise an error.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.converter = CastToType(drop_meta=drop_meta)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, dtype in self.key_iterator(d, self.dtype):
            d[key] = self.converter(d[key], dtype=dtype)

        return d


class ToTensord(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    backend = ToTensor.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        wrap_sequence: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: target data content type to convert, for example: torch.float, etc.
            device: specify the target device to put the Tensor data.
            wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
                E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToTensor(dtype=dtype, device=device, wrap_sequence=wrap_sequence)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # Create inverse transform
            inverse_transform = ToNumpy()
            # Apply inverse
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class EnsureTyped(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.EnsureType`.

    Ensure the input data to be a PyTorch Tensor or numpy array, support: `numpy array`, `PyTorch Tensor`,
    `float`, `int`, `bool`, `string` and `object` keep the original.
    If passing a dictionary, list or tuple, still return dictionary, list or tuple and recursively convert
    every item to the expected data type if `wrap_sequence=False`.

    Note: Currently, we only convert tensor data to numpy array or scalar number in the inverse operation.

    """

    backend = EnsureType.backend

    def __init__(
        self,
        keys: KeysCollection,
        data_type: str = "tensor",
        dtype: Union[DtypeLike, torch.dtype] = None,
        device: Optional[torch.device] = None,
        wrap_sequence: bool = True,
        drop_meta: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            data_type: target data type to convert, should be "tensor" or "numpy".
            dtype: target data content type to convert, for example: np.float32, torch.float, etc.
            device: for Tensor data type, specify the target device.
            wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
                E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.
            drop_meta: whether to drop the meta information of the input data, default to `True`.
                If `True`, then the meta information will be dropped quietly, unless the output type is MetaTensor.
                If `False`, converting a MetaTensor into a non-metatensor instance will raise an error.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = EnsureType(
            data_type=data_type, dtype=dtype, device=device, wrap_sequence=wrap_sequence, drop_meta=drop_meta
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            # FIXME: currently, only convert tensor data to numpy array or scalar number,
            # need to also invert numpy array but it's not easy to determine the previous data type
            d[key] = convert_to_numpy(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class ToNumpyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToNumpy`.
    """

    backend = ToNumpy.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = None,
        wrap_sequence: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: target data type when converting to numpy array.
            wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
                E.g., if `False`, `[1, 2]` -> `[array(1), array(2)]`, if `True`, then `[1, 2]` -> `array([1, 2])`.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToNumpy(dtype=dtype, wrap_sequence=wrap_sequence)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ToCupyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToCupy`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        dtype: data type specifier. It is inferred from the input by default.
            if not None, must be an argument of `numpy.dtype`, for more details:
            https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[array(1), array(2)]`, if `True`, then `[1, 2]` -> `array([1, 2])`.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ToCupy.backend

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Optional[np.dtype] = None,
        wrap_sequence: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = ToCupy(dtype=dtype, wrap_sequence=wrap_sequence)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ToPILd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToNumpy`.
    """

    backend = ToPIL.backend

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

    backend = Transpose.backend

    def __init__(
        self, keys: KeysCollection, indices: Optional[Sequence[int]], allow_missing_keys: bool = False
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = Transpose(indices)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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
            fwd_indices = np.array(transform[TraceKeys.EXTRA_INFO]["indices"])
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

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, keys: KeysCollection, sep: str = ".", use_re: Union[Sequence[bool], bool] = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to delete, can be "A{sep}B{sep}C"
                to delete key `C` in nested dictionary, `C` can be regular expression.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sep: the separator tag to define nested dictionary keys, default to ".".
            use_re: whether the specified key is a regular expression, it also can be
                a list of bool values, map the to keys.
        """
        super().__init__(keys)
        self.sep = sep
        self.use_re = ensure_tuple_rep(use_re, len(self.keys))

    def __call__(self, data):
        def _delete_item(keys, d, use_re: bool = False):
            key = keys[0]
            if len(keys) > 1:
                d[key] = _delete_item(keys[1:], d[key], use_re)
                return d
            return {k: v for k, v in d.items() if (use_re and not re.search(key, k)) or (not use_re and k != key)}

        d = dict(data)
        for key, use_re in zip(self.keys, self.use_re):
            d = _delete_item(key.split(self.sep), d, use_re)

        return d


class SelectItemsd(MapTransform):
    """
    Select only specified items from data dictionary to release memory.
    It will copy the selected key-values and construct and new dictionary.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, data):
        return {key: data[key] for key in self.key_iterator(data)}


class SqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SqueezeDim`.
    """

    backend = SqueezeDim.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class DataStatsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.DataStats`.
    """

    backend = DataStats.backend

    def __init__(
        self,
        keys: KeysCollection,
        prefix: Union[Sequence[str], str] = "Data",
        data_type: Union[Sequence[bool], bool] = True,
        data_shape: Union[Sequence[bool], bool] = True,
        value_range: Union[Sequence[bool], bool] = True,
        data_value: Union[Sequence[bool], bool] = False,
        additional_info: Optional[Union[Sequence[Callable], Callable]] = None,
        name: str = "DataStats",
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
            name: identifier of `logging.logger` to use, defaulting to "DataStats".
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.prefix = ensure_tuple_rep(prefix, len(self.keys))
        self.data_type = ensure_tuple_rep(data_type, len(self.keys))
        self.data_shape = ensure_tuple_rep(data_shape, len(self.keys))
        self.value_range = ensure_tuple_rep(value_range, len(self.keys))
        self.data_value = ensure_tuple_rep(data_value, len(self.keys))
        self.additional_info = ensure_tuple_rep(additional_info, len(self.keys))
        self.printer = DataStats(name=name)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, prefix, data_type, data_shape, value_range, data_value, additional_info in self.key_iterator(
            d, self.prefix, self.data_type, self.data_shape, self.value_range, self.data_value, self.additional_info
        ):
            d[key] = self.printer(d[key], prefix, data_type, data_shape, value_range, data_value, additional_info)
        return d


class SimulateDelayd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SimulateDelay`.
    """

    backend = SimulateDelay.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, delay_time in self.key_iterator(d, self.delay_time):
            d[key] = self.delayer(d[key], delay_time=delay_time)
        return d


class CopyItemsd(MapTransform):
    """
    Copy specified items from data dictionary and save with different key names.
    It can copy several items together and copy several times.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        times: int = 1,
        names: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            times: expected copy times, for example, if keys is "img", times is 3,
                it will add 3 copies of "img" data to the dictionary, default to 1.
            names: the names corresponding to the newly copied data,
                the length should match `len(keys) x times`. for example, if keys is ["img", "seg"]
                and times is 2, names can be: ["img_1", "seg_1", "img_2", "seg_2"].
                if None, use "{key}_{index}" as key for copy times `N`, index from `0` to `N-1`.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            ValueError: When ``times`` is nonpositive.
            ValueError: When ``len(names)`` is not ``len(keys) * times``. Incompatible values.

        """
        super().__init__(keys, allow_missing_keys)
        if times < 1:
            raise ValueError(f"times must be positive, got {times}.")
        self.times = times
        names = [f"{k}_{i}" for k in self.keys for i in range(self.times)] if names is None else ensure_tuple(names)
        if len(names) != (len(self.keys) * times):
            raise ValueError(
                "len(names) must match len(keys) * times, "
                f"got len(names)={len(names)} len(keys) * times={len(self.keys) * times}."
            )
        self.names = names

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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
                val = d[key]
                if isinstance(val, torch.Tensor):
                    d[new_key] = val.detach().clone()
                else:
                    d[new_key] = deepcopy(val)
        return d


class ConcatItemsd(MapTransform):
    """
    Concatenate specified items from data dictionary together on the first dim to construct a big array.
    Expect all the items are numpy array or PyTorch Tensor.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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

        if len(output) == 0:
            return d

        if data_type is np.ndarray:
            d[self.name] = np.concatenate(output, axis=self.dim)
        elif data_type is torch.Tensor:
            d[self.name] = torch.cat(output, dim=self.dim)  # type: ignore
        else:
            raise TypeError(f"Unsupported data type: {data_type}, available options are (numpy.ndarray, torch.Tensor).")
        return d


class Lambdad(MapTransform, InvertibleTransform):
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
        inv_func: Lambda/function of inverse operation if want to invert transforms, default to `lambda x: x`.
            It also can be a sequence of Callable, each element corresponds to a key in ``keys``.
        overwrite: whether to overwrite the original data in the input dictionary with lamdbda function output.
            default to True. it also can be a sequence of bool, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.

    Note: The inverse operation doesn't allow to define `extra_info` or access other information, such as the
        image's original size. If need these complicated information, please write a new InvertibleTransform directly.

    """

    backend = Lambda.backend

    def __init__(
        self,
        keys: KeysCollection,
        func: Union[Sequence[Callable], Callable],
        inv_func: Union[Sequence[Callable], Callable] = no_collation,
        overwrite: Union[Sequence[bool], bool] = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.func = ensure_tuple_rep(func, len(self.keys))
        self.inv_func = ensure_tuple_rep(inv_func, len(self.keys))
        self.overwrite = ensure_tuple_rep(overwrite, len(self.keys))
        self._lambd = Lambda()

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key, func, overwrite in self.key_iterator(d, self.func, self.overwrite):
            ret = self._lambd(img=d[key], func=func)
            if overwrite:
                d[key] = ret
            if isinstance(d[key], MetaTensor):
                self.push_transform(d, key)
        return d

    def inverse(self, data):
        d = deepcopy(dict(data))
        for key, overwrite in self.key_iterator(d, self.overwrite):
            if isinstance(d[key], MetaTensor):
                self.pop_transform(d[key])
            ret = self._lambd.inverse(data=d[key])
            if overwrite:
                d[key] = ret
        return d


class RandLambdad(Lambdad, RandomizableTransform):
    """
    Randomizable version :py:class:`monai.transforms.Lambdad`, the input `func` may contain random logic,
    or randomly execute the function based on `prob`. so `CacheDataset` will not execute it and cache the results.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        func: Lambda/function to be applied. It also can be a sequence of Callable,
            each element corresponds to a key in ``keys``.
        inv_func: Lambda/function of inverse operation if want to invert transforms, default to `lambda x: x`.
            It also can be a sequence of Callable, each element corresponds to a key in ``keys``.
        overwrite: whether to overwrite the original data in the input dictionary with lamdbda function output.
            default to True. it also can be a sequence of bool, each element corresponds to a key in ``keys``.
        prob: probability of executing the random function, default to 1.0, with 100% probability to execute.
            note that all the data specified by `keys` will share the same random probability to execute or not.
        allow_missing_keys: don't raise exception if key is missing.

    For more details, please check :py:class:`monai.transforms.Lambdad`.

    Note: The inverse operation doesn't allow to define `extra_info` or access other information, such as the
        image's original size. If need these complicated information, please write a new InvertibleTransform directly.
    """

    backend = Lambda.backend

    def __init__(
        self,
        keys: KeysCollection,
        func: Union[Sequence[Callable], Callable],
        inv_func: Union[Sequence[Callable], Callable] = no_collation,
        overwrite: Union[Sequence[bool], bool] = True,
        prob: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        Lambdad.__init__(
            self=self,
            keys=keys,
            func=func,
            inv_func=inv_func,
            overwrite=overwrite,
            allow_missing_keys=allow_missing_keys,
        )
        RandomizableTransform.__init__(self=self, prob=prob, do_transform=True)

    def __call__(self, data):
        self.randomize(data)
        d = dict(data)
        for key, func, overwrite in self.key_iterator(d, self.func, self.overwrite):
            if self._do_transform:
                ret = self._lambd(d[key], func=func)
                if overwrite:
                    d[key] = ret
            if isinstance(d[key], MetaTensor):
                self.push_transform(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = deepcopy(dict(data))
        for key, overwrite in self.key_iterator(d, self.overwrite):
            if isinstance(d[key], MetaTensor) and not self.pop_transform(d[key])[TraceKeys.DO_TRANSFORM]:
                continue
            ret = self._lambd.inverse(d[key])
            if overwrite:
                d[key] = ret
        return d


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

    backend = LabelToMask.backend

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        keys: KeysCollection,
        select_labels: Union[Sequence[int], int],
        merge_channels: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:  # pytype: disable=annotation-type-mismatch
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelToMask(select_labels=select_labels, merge_channels=merge_channels)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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

    backend = FgBgToIndices.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        image = d[self.image_key] if self.image_key else None
        for key in self.key_iterator(d):
            d[str(key) + self.fg_postfix], d[str(key) + self.bg_postfix] = self.converter(d[key], image)

        return d


class ClassesToIndicesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ClassesToIndices`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        indices_postfix: postfix to save the computed indices of all classes in dict.
            for example, if computed on `label` and `postfix = "_cls_indices"`, the key will be `label_cls_indices`.
        num_classes: number of classes for argmax label, not necessary for One-Hot label.
        image_key: if image_key is not None, use ``image > image_threshold`` to define valid region, and only select
            the indices within the valid region.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine the valid image content
            area and select only the indices of classes in this area.
        output_shape: expected shape of output indices. if not None, unravel indices to specified shape.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = ClassesToIndices.backend

    def __init__(
        self,
        keys: KeysCollection,
        indices_postfix: str = "_cls_indices",
        num_classes: Optional[int] = None,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        output_shape: Optional[Sequence[int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.indices_postfix = indices_postfix
        self.image_key = image_key
        self.converter = ClassesToIndices(num_classes, image_threshold, output_shape)

    def __call__(self, data: Mapping[Hashable, Any]):
        d = dict(data)
        image = d[self.image_key] if self.image_key else None
        for key in self.key_iterator(d):
            d[str(key) + self.indices_postfix] = self.converter(d[key], image)

        return d


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnBratsClasses.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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

    backend = AddExtremePointsChannel.backend

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

    def randomize(self, label: NdarrayOrTensor) -> None:
        self.points = get_extreme_points(label, rand_state=self.R, background=self.background, pert=self.pert)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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
            points_image, *_ = convert_to_dst_type(points_image, img)  # type: ignore
            d[key] = concatenate([img, points_image], axis=0)
        return d


class TorchVisiond(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.TorchVision` for non-randomized transforms.
    For randomized transforms of TorchVision use :py:class:`monai.transforms.RandTorchVisiond`.

    Note:
        As most of the TorchVision transforms only work for PIL image and PyTorch Tensor, this transform expects input
        data to be dict of PyTorch Tensors, users can easily call `ToTensord` transform to convert Numpy to Tensor.
    """

    backend = TorchVision.backend

    def __init__(self, keys: KeysCollection, name: str, allow_missing_keys: bool = False, *args, **kwargs) -> None:
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
        self.name = name
        self.trans = TorchVision(name, *args, **kwargs)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
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

    backend = TorchVision.backend

    def __init__(self, keys: KeysCollection, name: str, allow_missing_keys: bool = False, *args, **kwargs) -> None:
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
        self.name = name
        self.trans = TorchVision(name, *args, **kwargs)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.trans(d[key])
        return d


class MapLabelValued(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MapLabelValue`.
    """

    backend = MapLabelValue.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.mapper(d[key])
        return d


class IntensityStatsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.IntensityStats`.
    Compute statistics for the intensity values of input image and store into the metadata dictionary.
    For example: if `ops=[lambda x: np.mean(x), "max"]` and `key_prefix="orig"`, may generate below stats:
    `{"orig_custom_0": 1.5, "orig_max": 3.0}`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        ops: expected operations to compute statistics for the intensity.
            if a string, will map to the predefined operations, supported: ["mean", "median", "max", "min", "std"]
            mapping to `np.nanmean`, `np.nanmedian`, `np.nanmax`, `np.nanmin`, `np.nanstd`.
            if a callable function, will execute the function on input image.
        key_prefix: the prefix to combine with `ops` name to generate the key to store the results in the
            metadata dictionary. if some `ops` are callable functions, will use "{key_prefix}_custom_{index}"
            as the key, where index counts from 0.
        mask_keys: if not None, specify the mask array for the image to extract only the interested area to compute
            statistics, mask must have the same shape as the image.
            it should be a sequence of strings or None, map to the `keys`.
        channel_wise: whether to compute statistics for every channel of input image separately.
            if True, return a list of values for every operation, default to False.
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            used to store the computed statistics to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            used to store the computed statistics to the meta dict.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = IntensityStats.backend

    def __init__(
        self,
        keys: KeysCollection,
        ops: Sequence[Union[str, Callable]],
        key_prefix: str,
        mask_keys: Optional[KeysCollection] = None,
        channel_wise: bool = False,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.stats = IntensityStats(ops=ops, key_prefix=key_prefix, channel_wise=channel_wise)
        self.mask_keys = ensure_tuple_rep(None, len(self.keys)) if mask_keys is None else ensure_tuple(mask_keys)
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mask_key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.mask_keys, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            d[key], d[meta_key] = self.stats(
                img=d[key], meta_data=d.get(meta_key), mask=d.get(mask_key) if mask_key is not None else None
            )
        return d


class ToDeviced(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToDevice`.
    """

    backend = ToDevice.backend

    def __init__(
        self, keys: KeysCollection, device: Union[torch.device, str], allow_missing_keys: bool = False, **kwargs
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            device: target device to move the Tensor, for example: "cuda:1".
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: other args for the PyTorch `Tensor.to()` API, for more details:
                https://pytorch.org/docs/stable/generated/torch.Tensor.to.html.
        """
        super().__init__(keys, allow_missing_keys)
        self.converter = ToDevice(device=device, **kwargs)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class CuCIMd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CuCIM` for non-randomized transforms.
    For randomized transforms of CuCIM use :py:class:`monai.transforms.RandCuCIMd`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        name: The transform name in CuCIM package.
        allow_missing_keys: don't raise exception if key is missing.
        args: parameters for the CuCIM transform.
        kwargs: parameters for the CuCIM transform.

    Note:
        CuCIM transforms only work with CuPy arrays, this transform expects input data to be `cupy.ndarray`.
        Users can call `ToCuPy` transform to convert a numpy array or torch tensor to cupy array.
    """

    def __init__(self, keys: KeysCollection, name: str, allow_missing_keys: bool = False, *args, **kwargs) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.name = name
        self.trans = CuCIM(name, *args, **kwargs)

    def __call__(self, data):
        """
        Args:
            data: Dict[Hashable, `cupy.ndarray`]

        Returns:
            Dict[Hashable, `cupy.ndarray`]

        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.trans(d[key])
        return d


class RandCuCIMd(CuCIMd, RandomizableTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CuCIM` for randomized transforms.
    For deterministic non-randomized transforms of CuCIM use :py:class:`monai.transforms.CuCIMd`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        name: The transform name in CuCIM package.
        apply_prob: the probability to apply the transform (default=1.0)
        allow_missing_keys: don't raise exception if key is missing.
        args: parameters for the CuCIM transform.
        kwargs: parameters for the CuCIM transform.

    Note:
        - CuCIM transform only work with CuPy arrays, so this transform expects input data to be `cupy.ndarray`.
          Users can call `ToCuPy` transform to convert a numpy array or torch tensor to cupy array.
        - If the cuCIM transform is already randomized the `apply_prob` argument has nothing to do with
          the randomness of the underlying cuCIM transform. `apply_prob` defines if the transform (either randomized
          or non-randomized) being applied randomly, so it can apply non-randomized transforms randomly but be careful
          with setting `apply_prob` to anything than 1.0 when using along with cuCIM's randomized transforms.
        - If the random factor of the underlying cuCIM transform is not derived from `self.R`,
          the results may not be deterministic. See Also: :py:class:`monai.transforms.Randomizable`.
    """

    def __init__(self, apply_prob: float = 1.0, *args, **kwargs) -> None:
        CuCIMd.__init__(self, *args, **kwargs)
        RandomizableTransform.__init__(self, prob=apply_prob)

    def __call__(self, data):
        """
        Args:
            data: Dict[Hashable, `cupy.ndarray`]

        Returns:
            Dict[Hashable, `cupy.ndarray`]

        """
        self.randomize(data)
        if not self._do_transform:
            return dict(data)
        return super().__call__(data)


class AddCoordinateChannelsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddCoordinateChannels`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        spatial_dims: the spatial dimensions that are to have their coordinates encoded in a channel and
            appended to the input image. E.g., `(0, 1, 2)` represents `H, W, D` dims and append three channels
            to the input image, encoding the coordinates of the input's three spatial dimensions.
        allow_missing_keys: don't raise exception if key is missing.

    .. deprecated:: 0.8.0
        ``spatial_channels`` is deprecated, use ``spatial_dims`` instead.

    """

    backend = AddCoordinateChannels.backend

    @deprecated_arg(
        name="spatial_channels", new_name="spatial_dims", since="0.8", msg_suffix="please use `spatial_dims` instead."
    )
    def __init__(self, keys: KeysCollection, spatial_dims: Sequence[int], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.add_coordinate_channels = AddCoordinateChannels(spatial_dims=spatial_dims)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.add_coordinate_channels(d[key])
        return d


IdentityD = IdentityDict = Identityd
AsChannelFirstD = AsChannelFirstDict = AsChannelFirstd
AsChannelLastD = AsChannelLastDict = AsChannelLastd
AddChannelD = AddChannelDict = AddChanneld
EnsureChannelFirstD = EnsureChannelFirstDict = EnsureChannelFirstd
RemoveRepeatedChannelD = RemoveRepeatedChannelDict = RemoveRepeatedChanneld
RepeatChannelD = RepeatChannelDict = RepeatChanneld
SplitChannelD = SplitChannelDict = SplitChanneld
SplitDimD = SplitDimDict = SplitDimd
CastToTypeD = CastToTypeDict = CastToTyped
ToTensorD = ToTensorDict = ToTensord
EnsureTypeD = EnsureTypeDict = EnsureTyped
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
ClassesToIndicesD = ClassesToIndicesDict = ClassesToIndicesd
ConvertToMultiChannelBasedOnBratsClassesD = (
    ConvertToMultiChannelBasedOnBratsClassesDict
) = ConvertToMultiChannelBasedOnBratsClassesd
AddExtremePointsChannelD = AddExtremePointsChannelDict = AddExtremePointsChanneld
TorchVisionD = TorchVisionDict = TorchVisiond
RandTorchVisionD = RandTorchVisionDict = RandTorchVisiond
RandLambdaD = RandLambdaDict = RandLambdad
MapLabelValueD = MapLabelValueDict = MapLabelValued
IntensityStatsD = IntensityStatsDict = IntensityStatsd
ToDeviceD = ToDeviceDict = ToDeviced
CuCIMD = CuCIMDict = CuCIMd
RandCuCIMD = RandCuCIMDict = RandCuCIMd
AddCoordinateChannelsD = AddCoordinateChannelsDict = AddCoordinateChannelsd
