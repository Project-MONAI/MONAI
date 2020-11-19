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
A collection of "vanilla" transforms for utility functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import logging
import time
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch

from monai.transforms.compose import Transform
from monai.transforms.utils import map_binary_to_indices
from monai.utils import ensure_tuple

# Generic type which can represent either a numpy.ndarray or a torch.Tensor
# Unlike Union can create a dependence between parameter(s) / return(s)
NdarrayTensor = TypeVar("NdarrayTensor", np.ndarray, torch.Tensor)


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
        assert isinstance(channel_dim, int) and channel_dim >= -1, "invalid channel dimension."
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
        assert isinstance(channel_dim, int) and channel_dim >= -1, "invalid channel dimension."
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

    def __call__(self, img: NdarrayTensor) -> NdarrayTensor:
        """
        Apply the transform to `img`.
        """
        return img[None]


class RepeatChannel(Transform):
    """
    Repeat channel data to construct expected input shape for models.
    The `repeats` count includes the origin data, for example:
    ``RepeatChannel(repeats=2)([[1, 2], [3, 4]])`` generates: ``[[1, 2], [1, 2], [3, 4], [3, 4]]``

    Args:
        repeats: the number of repetitions for each element.
    """

    def __init__(self, repeats: int) -> None:
        assert repeats > 0, "repeats count must be greater than 0."
        self.repeats = repeats

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        return np.repeat(img, self.repeats, 0)


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

        outputs = list()
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

    def __init__(self, dtype: Union[np.dtype, torch.dtype] = np.float32) -> None:
        """
        Args:
            dtype: convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(
        self, img: Union[np.ndarray, torch.Tensor], dtype: Optional[Union[np.dtype, torch.dtype]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is a numpy array or PyTorch Tensor.

        Args:
            dtype: convert image to this data type, default is `self.dtype`.

        Raises:
            TypeError: When ``img`` type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        if isinstance(img, np.ndarray):
            return img.astype(self.dtype if dtype is None else dtype)
        elif torch.is_tensor(img):
            return torch.as_tensor(img, dtype=self.dtype if dtype is None else dtype)
        else:
            raise TypeError(f"img must be one of (numpy.ndarray, torch.Tensor) but is {type(img).__name__}.")


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply the transform to `img` and make it contiguous.
        """
        if torch.is_tensor(img):
            return img.contiguous()
        return torch.as_tensor(np.ascontiguousarray(img))


class ToNumpy(Transform):
    """
    Converts the input data to numpy array, can support list or tuple of numbers and PyTorch Tensor.
    """

    def __call__(self, img: Union[List, Tuple, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Apply the transform to `img` and make it contiguous.
        """
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()  # type: ignore
        return np.ascontiguousarray(img)


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
        return img.transpose(self.indices)


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
        return img.squeeze(self.dim)


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
        data_shape: bool = True,
        value_range: bool = True,
        data_value: bool = False,
        additional_info: Optional[Callable] = None,
        logger_handler: Optional[logging.Handler] = None,
    ) -> None:
        """
        Args:
            prefix: will be printed in format: "{prefix} statistics".
            data_shape: whether to show the shape of input data.
            value_range: whether to show the value range of input data.
            data_value: whether to show the raw value of input data.
                a typical example is to print some properties of Nifti image: affine, pixdim, etc.
            additional_info: user can define callable function to extract additional info from input data.
            logger_handler: add additional handler to output data: save to file, etc.
                add existing python logging handlers: https://docs.python.org/3/library/logging.handlers.html

        Raises:
            TypeError: When ``additional_info`` is not an ``Optional[Callable]``.

        """
        assert isinstance(prefix, str), "prefix must be a string."
        self.prefix = prefix
        self.data_shape = data_shape
        self.value_range = value_range
        self.data_value = data_value
        if additional_info is not None and not callable(additional_info):
            raise TypeError(f"additional_info must be None or callable but is {type(additional_info).__name__}.")
        self.additional_info = additional_info
        self.output: Optional[str] = None
        logging.basicConfig(level=logging.NOTSET)
        self._logger = logging.getLogger("DataStats")
        if logger_handler is not None:
            self._logger.addHandler(logger_handler)

    def __call__(
        self,
        img: NdarrayTensor,
        prefix: Optional[str] = None,
        data_shape: Optional[bool] = None,
        value_range: Optional[bool] = None,
        data_value: Optional[bool] = None,
        additional_info: Optional[Callable] = None,
    ) -> NdarrayTensor:
        """
        Apply the transform to `img`, optionally take arguments similar to the class constructor.
        """
        lines = [f"{prefix or self.prefix} statistics:"]

        if self.data_shape if data_shape is None else data_shape:
            lines.append(f"Shape: {img.shape}")
        if self.value_range if value_range is None else value_range:
            if isinstance(img, np.ndarray):
                lines.append(f"Value range: ({np.min(img)}, {np.max(img)})")
            elif torch.is_tensor(img):
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
        self._logger.debug(self.output)

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
        else:
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
    ) -> np.ndarray:
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
