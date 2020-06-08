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
from collections import OrderedDict

from typing import Callable, Optional, Iterable, Dict, Any
import logging
import numpy as np
import torch

from monai.transforms.compose import Transform
from monai.utils import validate_kwargs


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
        channel_dim (int): which dimension of input image is the channel, default is the last dimension.
    """

    def __init__(self, channel_dim: int = -1):
        assert isinstance(channel_dim, int) and channel_dim >= -1, "invalid channel dimension."
        self.channel_dim = channel_dim

    def __call__(self, img):
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
        channel_dim (int): which dimension of input image is the channel, default is the first dimension.
    """

    def __init__(self, channel_dim: int = 0):
        assert isinstance(channel_dim, int) and channel_dim >= -1, "invalid channel dimension."
        self.channel_dim = channel_dim

    def __call__(self, img):
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
        return img[None]


class RepeatChannel(Transform):
    """
    Repeat channel data to construct expected input shape for models.
    The `repeats` count includes the origin data, for example:
    ``RepeatChannel(repeats=2)([[1, 2], [3, 4]])`` generates: ``[[1, 2], [1, 2], [3, 4], [3, 4]]``

    Args:
        repeats (int): the number of repetitions for each element.
    """

    def __init__(self, repeats: int):
        assert repeats > 0, "repeats count must be greater than 0."
        self.repeats = repeats

    def __call__(self, img):
        return np.repeat(img, self.repeats, 0)


class CastToType(Transform):
    """
    Cast the image data to specified numpy data type.
    """

    def __init__(self, dtype: np.dtype = np.float32):
        """
        Args:
            dtype (np.dtype): convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(self, data: Iterable, *args, **kwargs):
        """
        Args:
            data (numpy.ndarray)
        """
        assert len(args) == 0, f"Invalid arguments provided {args}"
        assert len(kwargs) == 0, f"Invalid arguments provided {kwargs}"
        assert isinstance(data, np.ndarray), "image must be numpy array."
        return data.astype(self.dtype)


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img):
        if torch.is_tensor(img):
            return img.contiguous()
        return torch.as_tensor(np.ascontiguousarray(img))


class Transpose(Transform):
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, indices):
        self.indices = indices

    def __call__(self, img):
        return img.transpose(self.indices)


class SqueezeDim(Transform):
    """
    Squeeze a unitary dimension.
    """

    def __init__(self, dim: Optional[int] = 0):
        """
        Args:
            dim (int or None): dimension to be squeezed. Default = 0
                "None" works when the input is numpy array.
        """
        if dim is not None and not isinstance(dim, int):
            raise ValueError(f"Invalid channel dimension {dim}")
        self.dim = dim

    def __call__(self, data: Iterable, *args, **kwargs):
        """
        Args:
            data (numpy.ndarray): numpy arrays with required dimension `dim` removed
        """
        assert len(args) == 0, f"Invalid arguments provided {args}"
        assert len(kwargs) == 0, f"Invalid arguments provided {kwargs}"
        assert isinstance(data, (np.ndarray, torch.Tensor))
        return data.squeeze(self.dim)  # pytype: disable=attribute-error


class DataStats(Transform):
    """
    Utility transform to show the statistics of data for debug or analysis.
    It can be inserted into any place of a transform chain and check results of previous transforms.
    """

    def __init__(
        self,
        prefix: str = "Data",
        data_shape: bool = True,
        intensity_range: bool = True,
        data_value: bool = False,
        additional_info: Optional[Callable] = None,
        logger_handler: Optional[logging.Handler] = None,
    ):
        """
        Args:
            prefix (string): will be printed in format: "{prefix} statistics".
            data_shape (bool): whether to show the shape of input data.
            intensity_range (bool): whether to show the intensity value range of input data.
            data_value (bool): whether to show the raw value of input data.
                a typical example is to print some properties of Nifti image: affine, pixdim, etc.
            additional_info (Callable): user can define callable function to extract additional info from input data.
            logger_handler (logging.handler): add additional handler to output data: save to file, etc.
                add existing python logging handlers: https://docs.python.org/3/library/logging.handlers.html
        """
        assert isinstance(prefix, str), "prefix must be a string."
        self.prefix = prefix
        self.data_shape = data_shape
        self.intensity_range = intensity_range
        self.data_value = data_value
        if additional_info is not None and not callable(additional_info):
            raise ValueError("argument `additional_info` must be a callable.")
        self.additional_info = additional_info
        self.output: Optional[str] = None
        logging.basicConfig(level=logging.NOTSET)
        self._logger = logging.getLogger("DataStats")
        if logger_handler is not None:
            self._logger.addHandler(logger_handler)

    def __call__(self, data: Iterable, *args, **kwargs):
        """
        Args:
            data
            prefix (Optional[str]): will be printed in format: "{prefix} statistics". Default is ``self.prefix``.
            data_shape (Optional[bool]): whether to show the shape of input data. Default is ``self.data_shape``.
            intensity_range (Optional[bool]): whether to show the intensity value range of input data.
                Default is ``self.intensity_range``.
            data_value (Optional[bool]): whether to show the raw value of input data. Default is ``self.data_value``.
                A typical example is to print some properties of Nifti image: affine, pixdim, etc. Default is ``self.data_value``.
            additional_info (Optional[Callable]): user can define callable function to extract additional info from input data.
                Default is ``self.additional_info``.
        """
        reference_args: OrderedDict = OrderedDict(
            {"prefix": None, "data_shape": None, "intensity_range": None, "data_value": None, "additional_info": None}
        )
        produced_args: Dict[str, Any] = validate_kwargs(args, kwargs, reference_args)

        prefix: Optional[str] = produced_args["prefix"]
        data_shape: Optional[bool] = produced_args["data_shape"]
        intensity_range: Optional[bool] = produced_args["intensity_range"]
        data_value: Optional[bool] = produced_args["data_value"]
        additional_info: Optional[Callable] = produced_args["additional_info"]

        lines = [f"{prefix or self.prefix} statistics:"]

        if self.data_shape if data_shape is None else data_shape:
            lines.append(f"Shape: {data.shape}")  # type: ignore
        if self.intensity_range if intensity_range is None else intensity_range:
            lines.append(f"Intensity range: ({np.min(data)}, {np.max(data)})")
        if self.data_value if data_value is None else data_value:
            lines.append(f"Value: {data}")
        additional_info = self.additional_info if additional_info is None else additional_info
        if additional_info is not None:
            lines.append(f"Additional info: {additional_info(data)}")
        separator = "\n"
        self.output = f"{separator.join(lines)}"
        self._logger.debug(self.output)

        return data


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
            delay_time(float): The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        super().__init__()
        self.delay_time: float = delay_time

    def __call__(self, img, delay_time=None):
        time.sleep(self.delay_time if delay_time is None else delay_time)
        return img
