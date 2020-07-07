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

import time

from typing import Callable, Optional
import logging
import numpy as np
import torch

from monai.transforms.compose import Transform


class Identity(Transform):
    """
    Convert the input to an np.ndarray, if input data is np.ndarray or subclasses, return unchanged data.
    As the output value is same as input, it can be used as a testing tool to verify the transform chain,
    Compose or transform adaptor, etc.

    """

    def __call__(self, img):
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

    def __init__(self, channel_dim: int = -1):
        assert isinstance(channel_dim, int) and channel_dim >= -1, "invalid channel dimension."
        self.channel_dim = channel_dim

    def __call__(self, img):
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

    def __init__(self, channel_dim: int = 0):
        assert isinstance(channel_dim, int) and channel_dim >= -1, "invalid channel dimension."
        self.channel_dim = channel_dim

    def __call__(self, img):
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

    def __call__(self, img):
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

    def __init__(self, repeats: int):
        assert repeats > 0, "repeats count must be greater than 0."
        self.repeats = repeats

    def __call__(self, img):
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        return np.repeat(img, self.repeats, 0)


class CastToType(Transform):
    """
    Cast the image data to specified numpy data type.
    """

    def __init__(self, dtype: np.dtype = np.float32):
        """
        Args:
            dtype: convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(self, img: np.ndarray, dtype=None):
        """
        Apply the transform to `img`, assuming `img` is a numpy array.
        """
        assert isinstance(img, np.ndarray), "image must be numpy array."
        return img.astype(self.dtype if dtype is None else dtype)


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img):
        """
        Apply the transform to `img` and make it contiguous.
        """
        if torch.is_tensor(img):
            return img.contiguous()
        return torch.as_tensor(np.ascontiguousarray(img))


class ToNumpy(Transform):
    """
    Converts the input Tensor data to numpy array.
    """

    def __call__(self, img):
        """
        Apply the transform to `img` and make it contiguous.
        """
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        return np.ascontiguousarray(img)


class Transpose(Transform):
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, indices) -> None:
        self.indices = indices

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return img.transpose(self.indices)


class SqueezeDim(Transform):
    """
    Squeeze a unitary dimension.
    """

    def __init__(self, dim: Optional[int] = 0):
        """
        Args:
            dim: dimension to be squeezed. Default = 0
                "None" works when the input is numpy array.

        Raises:
            ValueError: Invalid channel dimension {dim}

        """
        if dim is not None and not isinstance(dim, int):
            raise ValueError(f"Invalid channel dimension {dim}")
        self.dim = dim

    def __call__(self, img):
        """
        Args:
            img (ndarray): numpy arrays with required dimension `dim` removed
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
    ):
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
            ValueError: argument `additional_info` must be a callable.

        """
        assert isinstance(prefix, str), "prefix must be a string."
        self.prefix = prefix
        self.data_shape = data_shape
        self.value_range = value_range
        self.data_value = data_value
        if additional_info is not None and not callable(additional_info):
            raise ValueError("argument `additional_info` must be a callable.")
        self.additional_info = additional_info
        self.output: Optional[str] = None
        logging.basicConfig(level=logging.NOTSET)
        self._logger = logging.getLogger("DataStats")
        if logger_handler is not None:
            self._logger.addHandler(logger_handler)

    def __call__(
        self,
        img,
        prefix: Optional[str] = None,
        data_shape: Optional[bool] = None,
        value_range: Optional[bool] = None,
        data_value: Optional[bool] = None,
        additional_info=None,
    ):
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

    def __init__(self, delay_time: float = 0.0):
        """
        Args:
            delay_time: The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        super().__init__()
        self.delay_time: float = delay_time

    def __call__(self, img, delay_time=None):
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
    """

    def __init__(self, func: Optional[Callable] = None) -> None:
        if func is not None and not callable(func):
            raise ValueError("func must be callable.")
        self.func = func

    def __call__(self, img, func=None):
        """
        Apply `self.func` to `img`.
        """
        if func is not None:
            if not callable(func):
                raise ValueError("func must be callable.")
            return func(img)
        if self.func is not None:
            return self.func(img)
        else:
            raise RuntimeError("neither func or self.func is callable.")
