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
A collection of "vanilla" transforms for utility functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import logging
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, NdarrayTensor
from monai.transforms.transform import Randomizable, Transform
from monai.transforms.utils import extreme_points_to_image, get_extreme_points, map_binary_to_indices
from monai.utils import ensure_tuple, min_version, optional_import

PILImageImage, has_pil = optional_import("PIL.Image", name="Image")
pil_image_fromarray, _ = optional_import("PIL.Image", name="fromarray")

__all__ = [
    "Identity",
    "AsChannelFirst",
    "AsChannelLast",
    "AddChannel",
    "EnsureChannelFirst",
    "RepeatChannel",
    "RemoveRepeatedChannel",
    "SplitChannel",
    "CastToType",
    "ToTensor",
    "ToNumpy",
    "Transpose",
    "SqueezeDim",
    "DataStats",
    "SimulateDelay",
    "Lambda",
    "LabelToMask",
    "FgBgToIndices",
    "ConvertToMultiChannelBasedOnBratsClasses",
    "AddExtremePointsChannel",
    "TorchVision",
    "MapLabelValue",
]


class Identity(Transform):
    """
    Convert the input to an np.ndarray, if input data is np.ndarray or subclasses, return unchanged data.
    As the output value is same as input, it can be used as a testing tool to verify the transform chain,
    Compose or transform adaptor, etc.

    """

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        return np.asanyarray(img)


class AsChannelFirst(Transform):
    """
    Change the channel dimension of the image to the first dimension.

    Most of the image transformations in ``monai.transforms``
    assume the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used to convert, for example, a channel-last image array in shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels) into the channel-first format,
    so that the multidimensional image array can be correctly interpreted by the other transforms.

    Args:
        channel_dim: which dimension of input image is the channel, default is the last dimension.
    """

    def __init__(self, channel_dim: int = -1) -> None:
        if not (isinstance(channel_dim, int) and channel_dim >= -1):
            raise AssertionError("invalid channel dimension.")
        self.channel_dim = channel_dim

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        return np.moveaxis(img, self.channel_dim, 0)


class AsChannelLast(Transform):
    """
    Change the channel dimension of the image to the last dimension.

    Some of other 3rd party transforms assume the input image is in the channel-last format with shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels).

    This transform could be used to convert, for example, a channel-first image array in shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]) into the channel-last format,
    so that MONAI transforms can construct a chain with other 3rd party transforms together.

    Args:
        channel_dim: which dimension of input image is the channel, default is the first dimension.
    """

    def __init__(self, channel_dim: int = 0) -> None:
        if not (isinstance(channel_dim, int) and channel_dim >= -1):
            raise AssertionError("invalid channel dimension.")
        self.channel_dim = channel_dim

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        return np.moveaxis(img, self.channel_dim, -1)


class AddChannel(Transform):
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """

    def __call__(self, img: NdarrayTensor):
        """
        Apply the transform to `img`.
        """
        return img[None]


class EnsureChannelFirst(Transform):
    """
    Automatically adjust or add the channel dimension of input data to ensure `channel_first` shape.
    It extracts the `original_channel_dim` info from provided meta_data dictionary.
    Typical values of `original_channel_dim` can be: "no_channel", 0, -1.
    Convert the data to `channel_first` based on the `original_channel_dim` information.

    """

    def __call__(self, img: np.ndarray, meta_dict: Optional[Dict] = None):
        """
        Apply the transform to `img`.
        """
        if not isinstance(meta_dict, dict):
            raise ValueError("meta_dict must be a dictionary data.")

        channel_dim = meta_dict.get("original_channel_dim", None)

        if channel_dim is None:
            raise ValueError("meta_dict must contain `original_channel_dim` information.")
        if channel_dim == "no_channel":
            return AddChannel()(img)
        return AsChannelFirst(channel_dim=channel_dim)(img)


class RepeatChannel(Transform):
    """
    Repeat channel data to construct expected input shape for models.
    The `repeats` count includes the origin data, for example:
    ``RepeatChannel(repeats=2)([[1, 2], [3, 4]])`` generates: ``[[1, 2], [1, 2], [3, 4], [3, 4]]``

    Args:
        repeats: the number of repetitions for each element.
    """

    def __init__(self, repeats: int) -> None:
        if repeats <= 0:
            raise AssertionError("repeats count must be greater than 0.")
        self.repeats = repeats

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        return np.repeat(img, self.repeats, 0)


class RemoveRepeatedChannel(Transform):
    """
    RemoveRepeatedChannel data to undo RepeatChannel
    The `repeats` count specifies the deletion of the origin data, for example:
    ``RemoveRepeatedChannel(repeats=2)([[1, 2], [1, 2], [3, 4], [3, 4]])`` generates: ``[[1, 2], [3, 4]]``

    Args:
        repeats: the number of repetitions to be deleted for each element.
    """

    def __init__(self, repeats: int) -> None:
        if repeats <= 0:
            raise AssertionError("repeats count must be greater than 0.")

        self.repeats = repeats

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        if np.shape(img)[0] < 2:
            raise AssertionError("Image must have more than one channel")

        return np.array(img[:: self.repeats, :])


class SplitChannel(Transform):
    """
    Split Numpy array or PyTorch Tensor data according to the channel dim.
    It can help applying different following transforms to different channels.
    Channel number must be greater than 1.

    Args:
        channel_dim: which dimension of input image is the channel, default to None
            to automatically select: if data is numpy array, channel_dim is 0 as
            `numpy array` is used in the pre transforms, if PyTorch Tensor, channel_dim
            is 1 as in most of the cases `Tensor` is uses in the post transforms.
    """

    def __init__(self, channel_dim: Optional[int] = None) -> None:
        self.channel_dim = channel_dim

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> List[Union[np.ndarray, torch.Tensor]]:
        if self.channel_dim is None:
            # automatically select the default channel dim based on data type
            if isinstance(img, torch.Tensor):
                channel_dim = 1
            else:
                channel_dim = 0
        else:
            channel_dim = self.channel_dim

        n_classes = img.shape[channel_dim]
        if n_classes <= 1:
            raise RuntimeError("input image does not contain multiple channels.")

        outputs = []
        slices = [slice(None)] * len(img.shape)
        for i in range(n_classes):
            slices[channel_dim] = slice(i, i + 1)
            outputs.append(img[tuple(slices)])

        return outputs


class CastToType(Transform):
    """
    Cast the Numpy data to specified numpy data type, or cast the PyTorch Tensor to
    specified PyTorch data type.
    """

    def __init__(self, dtype=np.float32) -> None:
        """
        Args:
            dtype: convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(
        self, img: Union[np.ndarray, torch.Tensor], dtype: Optional[Union[DtypeLike, torch.dtype]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is a numpy array or PyTorch Tensor.

        Args:
            dtype: convert image to this data type, default is `self.dtype`.

        Raises:
            TypeError: When ``img`` type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        if isinstance(img, np.ndarray):
            return img.astype(self.dtype if dtype is None else dtype)  # type: ignore
        if isinstance(img, torch.Tensor):
            return torch.as_tensor(img, dtype=self.dtype if dtype is None else dtype)
        raise TypeError(f"img must be one of (numpy.ndarray, torch.Tensor) but is {type(img).__name__}.")


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img) -> torch.Tensor:
        """
        Apply the transform to `img` and make it contiguous.
        """
        if isinstance(img, torch.Tensor):
            return img.contiguous()
        return torch.as_tensor(np.ascontiguousarray(img))


class ToNumpy(Transform):
    """
    Converts the input data to numpy array, can support list or tuple of numbers and PyTorch Tensor.
    """

    def __call__(self, img) -> np.ndarray:
        """
        Apply the transform to `img` and make it contiguous.
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()  # type: ignore
        return np.ascontiguousarray(img)


class ToPIL(Transform):
    """
    Converts the input image (in the form of NumPy array or PyTorch Tensor) to PIL image
    """

    def __call__(self, img):
        """
        Apply the transform to `img` and make it contiguous.
        """
        if isinstance(img, PILImageImage):
            return img
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        return pil_image_fromarray(img)


class Transpose(Transform):
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, indices: Optional[Sequence[int]]) -> None:
        self.indices = None if indices is None else tuple(indices)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        return img.transpose(self.indices)  # type: ignore


class SqueezeDim(Transform):
    """
    Squeeze a unitary dimension.
    """

    def __init__(self, dim: Optional[int] = 0) -> None:
        """
        Args:
            dim: dimension to be squeezed. Default = 0
                "None" works when the input is numpy array.

        Raises:
            TypeError: When ``dim`` is not an ``Optional[int]``.

        """
        if dim is not None and not isinstance(dim, int):
            raise TypeError(f"dim must be None or a int but is {type(dim).__name__}.")
        self.dim = dim

    def __call__(self, img: NdarrayTensor) -> NdarrayTensor:
        """
        Args:
            img: numpy arrays with required dimension `dim` removed
        """
        return img.squeeze(self.dim)  # type: ignore


class DataStats(Transform):
    """
    Utility transform to show the statistics of data for debug or analysis.
    It can be inserted into any place of a transform chain and check results of previous transforms.
    It support both `numpy.ndarray` and `torch.tensor` as input data,
    so it can be used in pre-processing and post-processing.
    """

    def __init__(
        self,
        prefix: str = "Data",
        data_type: bool = True,
        data_shape: bool = True,
        value_range: bool = True,
        data_value: bool = False,
        additional_info: Optional[Callable] = None,
        logger_handler: Optional[logging.Handler] = None,
    ) -> None:
        """
        Args:
            prefix: will be printed in format: "{prefix} statistics".
            data_type: whether to show the type of input data.
            data_shape: whether to show the shape of input data.
            value_range: whether to show the value range of input data.
            data_value: whether to show the raw value of input data.
                a typical example is to print some properties of Nifti image: affine, pixdim, etc.
            additional_info: user can define callable function to extract additional info from input data.
            logger_handler: add additional handler to output data: save to file, etc.
                add existing python logging handlers: https://docs.python.org/3/library/logging.handlers.html
                the handler should have a logging level of at least `INFO`.

        Raises:
            TypeError: When ``additional_info`` is not an ``Optional[Callable]``.

        """
        if not isinstance(prefix, str):
            raise AssertionError("prefix must be a string.")
        self.prefix = prefix
        self.data_type = data_type
        self.data_shape = data_shape
        self.value_range = value_range
        self.data_value = data_value
        if additional_info is not None and not callable(additional_info):
            raise TypeError(f"additional_info must be None or callable but is {type(additional_info).__name__}.")
        self.additional_info = additional_info
        self.output: Optional[str] = None
        self._logger = logging.getLogger("DataStats")
        self._logger.setLevel(logging.INFO)
        console = logging.StreamHandler(sys.stdout)  # always stdout
        console.setLevel(logging.INFO)
        self._logger.addHandler(console)
        if logger_handler is not None:
            self._logger.addHandler(logger_handler)

    def __call__(
        self,
        img: NdarrayTensor,
        prefix: Optional[str] = None,
        data_type: Optional[bool] = None,
        data_shape: Optional[bool] = None,
        value_range: Optional[bool] = None,
        data_value: Optional[bool] = None,
        additional_info: Optional[Callable] = None,
    ) -> NdarrayTensor:
        """
        Apply the transform to `img`, optionally take arguments similar to the class constructor.
        """
        lines = [f"{prefix or self.prefix} statistics:"]

        if self.data_type if data_type is None else data_type:
            lines.append(f"Type: {type(img)}")
        if self.data_shape if data_shape is None else data_shape:
            lines.append(f"Shape: {img.shape}")
        if self.value_range if value_range is None else value_range:
            if isinstance(img, np.ndarray):
                lines.append(f"Value range: ({np.min(img)}, {np.max(img)})")
            elif isinstance(img, torch.Tensor):
                lines.append(f"Value range: ({torch.min(img)}, {torch.max(img)})")
            else:
                lines.append(f"Value range: (not a PyTorch or Numpy array, type: {type(img)})")
        if self.data_value if data_value is None else data_value:
            lines.append(f"Value: {img}")
        additional_info = self.additional_info if additional_info is None else additional_info
        if additional_info is not None:
            lines.append(f"Additional info: {additional_info(img)}")
        separator = "\n"
        self.output = f"{separator.join(lines)}"
        self._logger.info(self.output)

        return img


class SimulateDelay(Transform):
    """
    This is a pass through transform to be used for testing purposes. It allows
    adding fake behaviors that are useful for testing purposes to simulate
    how large datasets behave without needing to test on large data sets.

    For example, simulating slow NFS data transfers, or slow network transfers
    in testing by adding explicit timing delays. Testing of small test data
    can lead to incomplete understanding of real world issues, and may lead
    to sub-optimal design choices.
    """

    def __init__(self, delay_time: float = 0.0) -> None:
        """
        Args:
            delay_time: The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        super().__init__()
        self.delay_time: float = delay_time

    def __call__(self, img: NdarrayTensor, delay_time: Optional[float] = None) -> NdarrayTensor:
        """
        Args:
            img: data remain unchanged throughout this transform.
            delay_time: The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        time.sleep(self.delay_time if delay_time is None else delay_time)
        return img


class Lambda(Transform):
    """
    Apply a user-defined lambda as a transform.

    For example:

    .. code-block:: python
        :emphasize-lines: 2

        image = np.ones((10, 2, 2))
        lambd = Lambda(func=lambda x: x[:4, :, :])
        print(lambd(image).shape)
        (4, 2, 2)

    Args:
        func: Lambda/function to be applied.

    Raises:
        TypeError: When ``func`` is not an ``Optional[Callable]``.

    """

    def __init__(self, func: Optional[Callable] = None) -> None:
        if func is not None and not callable(func):
            raise TypeError(f"func must be None or callable but is {type(func).__name__}.")
        self.func = func

    def __call__(self, img: Union[np.ndarray, torch.Tensor], func: Optional[Callable] = None):
        """
        Apply `self.func` to `img`.

        Args:
            func: Lambda/function to be applied. Defaults to `self.func`.

        Raises:
            TypeError: When ``func`` is not an ``Optional[Callable]``.
            ValueError: When ``func=None`` and ``self.func=None``. Incompatible values.

        """
        if func is not None:
            if not callable(func):
                raise TypeError(f"func must be None or callable but is {type(func).__name__}.")
            return func(img)
        if self.func is not None:
            return self.func(img)
        raise ValueError("Incompatible values: func=None and self.func=None.")


class LabelToMask(Transform):
    """
    Convert labels to mask for other tasks. A typical usage is to convert segmentation labels
    to mask data to pre-process images and then feed the images into classification network.
    It can support single channel labels or One-Hot labels with specified `select_labels`.
    For example, users can select `label value = [2, 3]` to construct mask data, or select the
    second and the third channels of labels to construct mask data.
    The output mask data can be a multiple channels binary data or a single channel binary
    data that merges all the channels.

    Args:
        select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
            is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
            `select_labels` is the expected channel indices.
        merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
            will return a single channel mask with binary data.

    """

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        select_labels: Union[Sequence[int], int],
        merge_channels: bool = False,
    ) -> None:  # pytype: disable=annotation-type-mismatch
        self.select_labels = ensure_tuple(select_labels)
        self.merge_channels = merge_channels

    def __call__(
        self, img: np.ndarray, select_labels: Optional[Union[Sequence[int], int]] = None, merge_channels: bool = False
    ):
        """
        Args:
            select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
                is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
                `select_labels` is the expected channel indices.
            merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
                will return a single channel mask with binary data.
        """
        if select_labels is None:
            select_labels = self.select_labels
        else:
            select_labels = ensure_tuple(select_labels)

        if img.shape[0] > 1:
            data = img[[*select_labels]]
        else:
            data = np.where(np.in1d(img, select_labels), True, False).reshape(img.shape)

        return np.any(data, axis=0, keepdims=True) if (merge_channels or self.merge_channels) else data


class FgBgToIndices(Transform):
    def __init__(self, image_threshold: float = 0.0, output_shape: Optional[Sequence[int]] = None) -> None:
        """
        Compute foreground and background of the input label data, return the indices.
        If no output_shape specified, output data will be 1 dim indices after flattening.
        This transform can help pre-compute foreground and background regions for other transforms.
        A typical usage is to randomly select foreground and background to crop.
        The main logic is based on :py:class:`monai.transforms.utils.map_binary_to_indices`.

        Args:
            image_threshold: if enabled `image` at runtime, use ``image > image_threshold`` to
                determine the valid image content area and select background only in this area.
            output_shape: expected shape of output indices. if not None, unravel indices to specified shape.

        """
        self.image_threshold = image_threshold
        self.output_shape = output_shape

    def __call__(
        self,
        label: np.ndarray,
        image: Optional[np.ndarray] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            label: input data to compute foreground and background indices.
            image: if image is not None, use ``label = 0 & image > image_threshold``
                to define background. so the output items will not map to all the voxels in the label.
            output_shape: expected shape of output indices. if None, use `self.output_shape` instead.

        """
        if output_shape is None:
            output_shape = self.output_shape
        fg_indices, bg_indices = map_binary_to_indices(label, image, self.image_threshold)
        if output_shape is not None:
            fg_indices = np.stack([np.unravel_index(i, output_shape) for i in fg_indices])
            bg_indices = np.stack([np.unravel_index(i, output_shape) for i in bg_indices])

        return fg_indices, bg_indices


class ConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)

        result = []
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) to TC
        result.append(np.logical_or(img == 1, img == 4))
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
        result.append(np.logical_or(np.logical_or(img == 1, img == 4), img == 2))
        # label 4 is ET
        result.append(img == 4)
        return np.stack(result, axis=0)


class AddExtremePointsChannel(Randomizable):
    """
    Add extreme points of label to the image as a new channel. This transform generates extreme
    point from label and applies a gaussian filter. The pixel values in points image are rescaled
    to range [rescale_min, rescale_max] and added as a new channel to input image. The algorithm is
    described in Roth et al., Going to Extremes: Weakly Supervised Medical Image Segmentation
    https://arxiv.org/abs/2009.11988.

    This transform only supports single channel labels (1, spatial_dim1, [spatial_dim2, ...]). The
    background ``index`` is ignored when calculating extreme points.

    Args:
        background: Class index of background label, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.

    Raises:
        ValueError: When no label image provided.
        ValueError: When label image is not single channel.
    """

    def __init__(self, background: int = 0, pert: float = 0.0) -> None:
        self._background = background
        self._pert = pert
        self._points: List[Tuple[int, ...]] = []

    def randomize(self, label: np.ndarray) -> None:
        self._points = get_extreme_points(label, rand_state=self.R, background=self._background, pert=self._pert)

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 3.0,
        rescale_min: float = -1.0,
        rescale_max: float = 1.0,
    ):
        """
        Args:
            img: the image that we want to add new channel to.
            label: label image to get extreme points from. Shape must be
                (1, spatial_dim1, [, spatial_dim2, ...]). Doesn't support one-hot labels.
            sigma: if a list of values, must match the count of spatial dimensions of input data,
                and apply every value in the list to 1 spatial dimension. if only 1 value provided,
                use it for all spatial dimensions.
            rescale_min: minimum value of output data.
            rescale_max: maximum value of output data.
        """
        if label is None:
            raise ValueError("This transform requires a label array!")
        if label.shape[0] != 1:
            raise ValueError("Only supports single channel labels!")

        # Generate extreme points
        self.randomize(label[0, :])

        points_image = extreme_points_to_image(
            points=self._points, label=label, sigma=sigma, rescale_min=rescale_min, rescale_max=rescale_max
        )

        return np.concatenate([img, points_image], axis=0)


class TorchVision:
    """
    This is a wrapper transform for PyTorch TorchVision transform based on the specified transform name and args.
    As most of the TorchVision transforms only work for PIL image and PyTorch Tensor, this transform expects input
    data to be PyTorch Tensor, users can easily call `ToTensor` transform to convert a Numpy array to Tensor.

    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: The transform name in TorchVision package.
            args: parameters for the TorchVision transform.
            kwargs: parameters for the TorchVision transform.

        """
        super().__init__()
        transform, _ = optional_import("torchvision.transforms", "0.8.0", min_version, name=name)
        self.trans = transform(*args, **kwargs)

    def __call__(self, img: torch.Tensor):
        """
        Args:
            img: PyTorch Tensor data for the TorchVision transform.

        """
        return self.trans(img)


class MapLabelValue:
    """
    Utility to map label values to another set of values.
    For example, map [3, 2, 1] to [0, 1, 2], [1, 2, 3] -> [0.5, 1.5, 2.5], ["label3", "label2", "label1"] -> [0, 1, 2],
    [3.5, 2.5, 1.5] -> ["label0", "label1", "label2"], etc.
    The label data must be numpy array or array-like data and the output data will be numpy array.

    """

    def __init__(self, orig_labels: Sequence, target_labels: Sequence, dtype: DtypeLike = np.float32) -> None:
        """
        Args:
            orig_labels: original labels that map to others.
            target_labels: expected label values, 1: 1 map to the `orig_labels`.
            dtype: convert the output data to dtype, default to float32.

        """
        if len(orig_labels) != len(target_labels):
            raise ValueError("orig_labels and target_labels must have the same length.")
        if all([o == z for o, z in zip(orig_labels, target_labels)]):
            raise ValueError("orig_labels and target_labels are exactly the same, should be different to map.")

        self.orig_labels = orig_labels
        self.target_labels = target_labels
        self.dtype = dtype

    def __call__(self, img: np.ndarray):
        img = np.asarray(img)
        img_flat = img.flatten()
        try:
            out_flat = np.copy(img_flat).astype(self.dtype)
        except ValueError:
            # can't copy unchanged labels as the expected dtype is not supported, must map all the label values
            out_flat = np.zeros(shape=img_flat.shape, dtype=self.dtype)

        for o, t in zip(self.orig_labels, self.target_labels):
            if o == t:
                continue
            np.place(out_flat, img_flat == o, t)

        return out_flat.reshape(img.shape)
