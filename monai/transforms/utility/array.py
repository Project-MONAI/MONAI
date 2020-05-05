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

import numpy as np
import torch

from monai.transforms.compose import Transform


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

    def __init__(self, channel_dim=-1):
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

    def __init__(self, channel_dim=0):
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

    def __init__(self, repeats):
        assert repeats > 0, "repeats count must be greater than 0."
        self.repeats = repeats

    def __call__(self, img):
        return np.repeat(img, self.repeats, 0)


class CastToType(Transform):
    """
    Cast the image data to specified numpy data type.
    """

    def __init__(self, dtype=np.float32):
        """
        Args:
            dtype (np.dtype): convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(self, img):
        assert isinstance(img, np.ndarray), "image must be numpy array."
        return img.astype(self.dtype)


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
    Squeeze undesired unitary dimensions
    """

    def __init__(self, dim=None):
        """
        Args:
            dim (int): dimension to be squeezed.
                Default: None (all dimensions of size 1 will be removed)
        """
        if dim is not None:
            assert isinstance(dim, int) and dim >= -1, "invalid channel dimension."
        self.dim = dim

    def __call__(self, img):
        """
        Args:
            img (ndarray): numpy arrays with required dimension `dim` removed
        """
        return np.squeeze(img, self.dim)


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

    def __init__(self, delay_time=0.0):
        """
        Args:
            delay_time(float): The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        super().__init__()
        self.delay_time: float = delay_time

    def __call__(self, img):
        time.sleep(self.delay_time)
        return img
