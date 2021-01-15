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
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.utility.array import (
    AddChannel,
    AsChannelFirst,
    AsChannelLast,
    CastToType,
    ConvertToMultiChannelBasedOnBratsClasses,
    DataStats,
    FgBgToIndices,
    Identity,
    LabelToMask,
    Lambda,
    RepeatChannel,
    SimulateDelay,
    SplitChannel,
    SqueezeDim,
    ToNumpy,
    TorchVision,
    ToTensor,
)
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.utils import ensure_tuple, ensure_tuple_rep

__all__ = [
    "Identityd",
    "AsChannelFirstd",
    "AsChannelLastd",
    "AddChanneld",
    "RepeatChanneld",
    "SplitChanneld",
    "CastToTyped",
    "ToTensord",
    "ToNumpyd",
    "DeleteItemsd",
    "SelectItemsd",
    "SqueezeDimd",
    "DataStatsd",
    "SimulateDelayd",
    "CopyItemsd",
    "ConcatItemsd",
    "Lambdad",
    "LabelToMaskd",
    "FgBgToIndicesd",
    "ConvertToMultiChannelBasedOnBratsClassesd",
    "AddExtremePointsChanneld",
    "IdentityD",
    "IdentityDict",
    "AsChannelFirstD",
    "AsChannelFirstDict",
    "AsChannelLastD",
    "AsChannelLastDict",
    "AddChannelD",
    "AddChannelDict",
    "RepeatChannelD",
    "RepeatChannelDict",
    "SplitChannelD",
    "SplitChannelDict",
    "CastToTypeD",
    "CastToTypeDict",
    "ToTensorD",
    "ToTensorDict",
    "DeleteItemsD",
    "DeleteItemsDict",
    "SqueezeDimD",
    "SqueezeDimDict",
    "DataStatsD",
    "DataStatsDict",
    "SimulateDelayD",
    "SimulateDelayDict",
    "CopyItemsD",
    "CopyItemsDict",
    "ConcatItemsD",
    "ConcatItemsDict",
    "LambdaD",
    "LambdaDict",
    "LabelToMaskD",
    "LabelToMaskDict",
    "FgBgToIndicesD",
    "FgBgToIndicesDict",
    "ConvertToMultiChannelBasedOnBratsClassesD",
    "ConvertToMultiChannelBasedOnBratsClassesDict",
    "AddExtremePointsChannelD",
    "AddExtremePointsChannelDict",
    "TorchVisiond",
    "TorchVisionD",
    "TorchVisionDict",
]


class Identityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Identity`.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`

        """
        super().__init__(keys)
        self.identity = Identity()

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.identity(d[key])
        return d


class AsChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelFirst`.
    """

    def __init__(self, keys: KeysCollection, channel_dim: int = -1) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim: which dimension of input image is the channel, default is the last dimension.
        """
        super().__init__(keys)
        self.converter = AsChannelFirst(channel_dim=channel_dim)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AsChannelLastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsChannelLast`.
    """

    def __init__(self, keys: KeysCollection, channel_dim: int = 0) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim: which dimension of input image is the channel, default is the first dimension.
        """
        super().__init__(keys)
        self.converter = AsChannelLast(channel_dim=channel_dim)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.adder = AddChannel()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.adder(d[key])
        return d


class RepeatChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RepeatChannel`.
    """

    def __init__(self, keys: KeysCollection, repeats: int) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            repeats: the number of repetitions for each element.
        """
        super().__init__(keys)
        self.repeater = RepeatChannel(repeats)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
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

        """
        super().__init__(keys)
        self.output_postfixes = output_postfixes
        self.splitter = SplitChannel(channel_dim=channel_dim)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
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
        dtype: Union[Sequence[Union[np.dtype, torch.dtype]], np.dtype, torch.dtype] = np.float32,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of np.dtype or torch.dtype,
                each element corresponds to a key in ``keys``.

        """
        MapTransform.__init__(self, keys)
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.converter = CastToType()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key], dtype=self.dtype[idx])

        return d


class ToTensord(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToTensor`.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.converter = ToTensor()

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class ToNumpyd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ToNumpy`.
    """

    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.converter = ToNumpy()

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class DeleteItemsd(MapTransform):
    """
    Delete specified items from data dictionary to release memory.
    It will remove the key-values and copy the others to construct a new dictionary.
    """

    def __call__(self, data):
        return {key: val for key, val in data.items() if key not in self.keys}


class SelectItemsd(MapTransform):
    """
    Select only specified items from data dictionary to release memory.
    It will copy the selected key-values and construct and new dictionary.
    """

    def __call__(self, data):
        result = {key: val for key, val in data.items() if key in self.keys}
        return result


class SqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SqueezeDim`.
    """

    def __init__(self, keys: KeysCollection, dim: int = 0) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dim: dimension to be squeezed. Default: 0 (the first dimension)
        """
        super().__init__(keys)
        self.converter = SqueezeDim(dim=dim)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
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
        logger_handler: Optional[logging.Handler] = None,
    ) -> None:
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

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
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
    Dictionary-based wrapper of :py:class:`monai.transforms.SimulateDelay`.
    """

    def __init__(self, keys: KeysCollection, delay_time: Union[Sequence[float], float] = 0.0) -> None:
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

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.delayer(d[key], delay_time=self.delay_time[idx])
        return d


class CopyItemsd(MapTransform):
    """
    Copy specified items from data dictionary and save with different key names.
    It can copy several items together and copy several times.

    """

    def __init__(self, keys: KeysCollection, times: int, names: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            times: expected copy times, for example, if keys is "img", times is 3,
                it will add 3 copies of "img" data to the dictionary.
            names: the names corresponding to the newly copied data,
                the length should match `len(keys) x times`. for example, if keys is ["img", "seg"]
                and times is 2, names can be: ["img_1", "seg_1", "img_2", "seg_2"].

        Raises:
            ValueError: When ``times`` is nonpositive.
            ValueError: When ``len(names)`` is not ``len(keys) * times``. Incompatible values.

        """
        super().__init__(keys)
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
        for key, new_key in zip(self.keys * self.times, self.names):
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

    def __init__(self, keys: KeysCollection, name: str, dim: int = 0) -> None:
        """
        Args:
            keys: keys of the corresponding items to be concatenated together.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name corresponding to the key to store the concatenated data.
            dim: on which dimension to concatenate the items, default is 0.

        Raises:
            ValueError: When insufficient keys are given (``len(self.keys) < 2``).

        """
        super().__init__(keys)
        if len(self.keys) < 2:
            raise ValueError("Concatenation requires at least 2 keys.")
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
        for key in self.keys:
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
    """

    def __init__(self, keys: KeysCollection, func: Union[Sequence[Callable], Callable]) -> None:
        super().__init__(keys)
        self.func = ensure_tuple_rep(func, len(self.keys))
        self.lambd = Lambda()

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.lambd(d[key], func=self.func[idx])

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

    """

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        keys: KeysCollection,
        select_labels: Union[Sequence[int], int],
        merge_channels: bool = False,
    ) -> None:  # pytype: disable=annotation-type-mismatch
        super().__init__(keys)
        self.converter = LabelToMask(select_labels=select_labels, merge_channels=merge_channels)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
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

    """

    def __init__(
        self,
        keys: KeysCollection,
        fg_postfix: str = "_fg_indices",
        bg_postfix: str = "_bg_indices",
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        output_shape: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(keys)
        self.fg_postfix = fg_postfix
        self.bg_postfix = bg_postfix
        self.image_key = image_key
        self.converter = FgBgToIndices(image_threshold, output_shape)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        image = d[self.image_key] if self.image_key else None
        for key in self.keys:
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

    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        self.converter = ConvertToMultiChannelBasedOnBratsClasses()

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
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
    ):
        super().__init__(keys)
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

        for key in data.keys():
            if key in self.keys:
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
    Dictionary-based wrapper of :py:class:`monai.transforms.TorchVision`.
    As most of the TorchVision transforms only work for PIL image and PyTorch Tensor, this transform expects input
    data to be dict of PyTorch Tensors, users can easily call `ToTensord` transform to convert Numpy to Tensor.
    """

    def __init__(self, keys: KeysCollection, name: str, *args, **kwargs) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: The transform name in TorchVision package.
            args: parameters for the TorchVision transform.
            kwargs: parameters for the TorchVision transform.

        """
        super().__init__(keys)
        self.trans = TorchVision(name, *args, **kwargs)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.trans(d[key])
        return d


IdentityD = IdentityDict = Identityd
AsChannelFirstD = AsChannelFirstDict = AsChannelFirstd
AsChannelLastD = AsChannelLastDict = AsChannelLastd
AddChannelD = AddChannelDict = AddChanneld
RepeatChannelD = RepeatChannelDict = RepeatChanneld
SplitChannelD = SplitChannelDict = SplitChanneld
CastToTypeD = CastToTypeDict = CastToTyped
ToTensorD = ToTensorDict = ToTensord
DeleteItemsD = DeleteItemsDict = DeleteItemsd
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
