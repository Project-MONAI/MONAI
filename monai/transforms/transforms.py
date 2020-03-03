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
A collection of "vanilla" transforms
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import numpy as np
import torch

import monai
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms.compose import Randomizable
from monai.transforms.utils import rescale_array

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
class Flip:
    """Reverses the order of elements along the given axis. Preserves shape.
    Uses np.flip in practice. See numpy.flip for additional details.

    Args:
        axes (None, int or tuple of ints): Axes along which to flip over. Default is None.
    """

    def __init__(self, axis=None):
        assert axis is None or isinstance(axis, (int, list, tuple)), \
            "axis must be None, int or tuple of ints."
        self.axis = axis

    def __call__(self, img):
        return np.flip(img, self.axis)


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


@export
class ImageEndPadder:
    """Performs padding by appending to the end of the data all on one side for each dimension.
     Uses np.pad so in practice, a mode needs to be provided. See numpy.lib.arraypad.pad
     for additional details.

    Args:
        out_size (list): the size of region of interest at the end of the operation.
        mode (string): a portion from numpy.lib.arraypad.pad is copied below.
        dtype: output data format.
    """

    def __init__(self, out_size, mode, dtype=np.float32):
        assert out_size is not None and isinstance(out_size, (list, tuple)), 'out_size must be list or tuple'
        self.out_size = out_size
        assert isinstance(mode, str), 'mode must be str'
        self.mode = mode
        self.dtype = dtype

    def _determine_data_pad_width(self, data_shape):
        return [(0, max(self.out_size[i] - data_shape[i], 0)) for i in range(len(self.out_size))]

    def __call__(self, img):
        data_pad_width = self._determine_data_pad_width(img.shape[2:])
        all_pad_width = [(0, 0), (0, 0)] + data_pad_width
        img = np.pad(img, all_pad_width, self.mode)
        return img


@export
class Rotate90:
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    """

    def __init__(self, k=1, axes=(1, 2)):
        """
        Args:
            k (int): number of times to rotate by 90 degrees.
            axes (2 ints): defines the plane to rotate with 2 axes.
        """
        self.k = k
        self.plane_axes = axes

    def __call__(self, img):
        return np.rot90(img, self.k, self.plane_axes)


@export
class RandRotate90(Randomizable):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `axes`.
    """

    def __init__(self, prob=0.1, max_k=3, axes=(1, 2)):
        """
        Args:
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            axes (2 ints): defines the plane to rotate with 2 axes.
                (Default (1, 2))
        """
        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.axes = axes

        self._do_transform = False
        self._rand_k = 0

    def randomise(self):
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, img):
        self.randomise()
        if not self._do_transform:
            return img
        rotator = Rotate90(self._rand_k, self.axes)
        return rotator(img)


# if __name__ == "__main__":
#     img = np.array((1, 2, 3, 4)).reshape((1, 2, 2))
#     rotator = RandRotate90(prob=0.0, max_k=3, axes=(1, 2))
#     # rotator.set_random_state(1234)
#     img_result = rotator(img)
#     print(type(img))
#     print(img_result)
