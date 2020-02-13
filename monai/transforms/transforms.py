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


import torch
import numpy as np

import monai
from monai.transforms.utils import rescale_array
from monai.data.utils import get_random_patch, get_valid_patch_size

export = monai.utils.export("monai.transforms")


@export
class AddChannel:
    """
    Adds a 1-length channel dimension to the input image.
    """

    def __call__(self, img):
        return img[None]


@export
class Transpose:
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, indices):
        self.indices = indices

    def __call__(self, img):
        return img.transpose(self.indices)


@export
class Rescale:
    """
    Rescales the input image to the given value range.
    """

    def __init__(self, minv=0.0, maxv=1.0, dtype=np.float32):
        self.minv = minv
        self.maxv = maxv
        self.dtype = dtype

    def __call__(self, img):
        return rescale_array(img, self.minv, self.maxv, self.dtype)


@export
class ToTensor:
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img):
        return torch.from_numpy(img)


@export
class UniformRandomPatch:
    """
    Selects a patch of the given size chosen at a uniformly random position in the image.
    """

    def __init__(self, patch_size):
        self.patch_size = (None,) + tuple(patch_size)

    def __call__(self, img):
        patch_size = get_valid_patch_size(img.shape, self.patch_size)
        slices = get_random_patch(img.shape, patch_size)

        return img[slices]


@export
class IntensityNormalizer:
    """Normalize input based on provided args, using calculated mean and std if not provided
    (shape of subtrahend and divisor must match. if 0, entire volume uses same subtrahend and
     divisor, otherwise the shape can have dimension 1 for channels).
     Current implementation can only support 'channel_last' format data.

    Args:
        subtrahend (ndarray): the amount to subtract by (usually the mean)
        divisor (ndarray): the amount to divide by (usually the standard deviation)
        dtype: output data format
    """

    def __init__(self, subtrahend=None, divisor=None, dtype=np.float32):
        if subtrahend is not None or divisor is not None:
            assert isinstance(subtrahend, np.ndarray) and isinstance(divisor, np.ndarray), \
                'subtrahend and divisor must be set in pair and in numpy array.'
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.dtype = dtype

    def __call__(self, img):
        if self.subtrahend is not None and self.divisor is not None:
            img -= self.subtrahend
            img /= self.divisor
        else:
            img -= np.mean(img)
            img /= np.std(img)

        if self.dtype != img.dtype:
            img = img.astype(self.dtype)
        return img
