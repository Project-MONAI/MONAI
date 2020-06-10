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
A collection of "vanilla" transforms for intensity adjustment
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Optional, Tuple

import numpy as np

from monai.transforms.compose import Transform, Randomizable
from monai.transforms.utils import rescale_array


class RandGaussianNoise(Randomizable, Transform):
    """Add Gaussian noise to image.

    Args:
        prob: Probability to add Gaussian noise.
        mean (float or array of floats): Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
    """

    def __init__(self, prob: float = 0.1, mean=0.0, std: float = 0.1):
        self.prob = prob
        self.mean = mean
        self.std = std
        self._do_transform = False
        self._noise = None

    def randomize(self, im_shape) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random() < self.prob
        self._noise = self.R.normal(self.mean, self.R.uniform(0, self.std), size=im_shape)

    def __call__(self, img):
        self.randomize(img.shape)
        return img + self._noise.astype(img.dtype) if self._do_transform else img


class ShiftIntensity(Transform):
    """Shift intensity uniformly for the entire image with specified `offset`.

    Args:
        offset: offset value to shift the intensity of image.
    """

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, img):
        return (img + self.offset).astype(img.dtype)


class RandShiftIntensity(Randomizable, Transform):
    """Randomly shift intensity with randomly picked offset.
    """

    def __init__(self, offsets, prob: float = 0.1):
        """
        Args:
            offsets(int, float, tuple or list): offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            prob: probability of shift.
        """
        self.offsets = (-offsets, offsets) if not isinstance(offsets, (list, tuple)) else offsets
        assert len(self.offsets) == 2, "offsets should be a number or pair of numbers."
        self.prob = prob
        self._do_transform = False

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._offset = self.R.uniform(low=self.offsets[0], high=self.offsets[1])
        self._do_transform = self.R.random() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        shifter = ShiftIntensity(self._offset)
        return shifter(img)


class ScaleIntensity(Transform):
    """
    Scale the intensity of input image to the given value range (minv, maxv).
    If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
    """

    def __init__(
        self, minv: Optional[float] = 0.0, maxv: Optional[float] = 1.0, factor: Optional[float] = None
    ) -> None:
        """
        Args:
            minv: minimum value of output data.
            maxv: maximum value of output data.
            factor: factor scale by ``v = v * (1 + factor)``.
        """
        self.minv = minv
        self.maxv = maxv
        self.factor = factor

    def __call__(self, img):
        if self.minv is not None and self.maxv is not None:
            return rescale_array(img, self.minv, self.maxv, img.dtype)
        else:
            return (img * (1 + self.factor)).astype(img.dtype)


class RandScaleIntensity(Randomizable, Transform):
    """
    Randomly scale the intensity of input image by ``v = v * (1 + factor)`` where the `factor`
    is randomly picked from (factors[0], factors[0]).
    """

    def __init__(self, factors, prob: float = 0.1):
        """
        Args:
            factors(float, tuple or list): factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of scale.

        """
        self.factors = (-factors, factors) if not isinstance(factors, (list, tuple)) else factors
        assert len(self.factors) == 2, "factors should be a number or pair of numbers."
        self.prob = prob
        self._do_transform = False

    def randomize(self) -> None:  # type: ignore # see issue #495
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])
        self._do_transform = self.R.random() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        scaler = ScaleIntensity(minv=None, maxv=None, factor=self.factor)
        return scaler(img)


class NormalizeIntensity(Transform):
    """Normalize input based on provided args, using calculated mean and std if not provided
    (shape of subtrahend and divisor must match. if 0, entire volume uses same subtrahend and
    divisor, otherwise the shape can have dimension 1 for channels).
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        subtrahend (ndarray): the amount to subtract by (usually the mean).
        divisor (ndarray): the amount to divide by (usually the standard deviation).
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
        self,
        subtrahend: Optional[np.ndarray] = None,
        divisor: Optional[np.ndarray] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
    ):
        if subtrahend is not None or divisor is not None:
            assert isinstance(subtrahend, np.ndarray) and isinstance(
                divisor, np.ndarray
            ), "subtrahend and divisor must be set in pair and in numpy array."
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def _normalize(self, img):
        slices = (img != 0) if self.nonzero else np.ones(img.shape, dtype=np.bool_)
        if np.any(slices):
            if self.subtrahend is not None and self.divisor is not None:
                img[slices] = (img[slices] - self.subtrahend[slices]) / self.divisor[slices]
            else:
                img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
        return img

    def __call__(self, img):
        if self.channel_wise:
            for i, d in enumerate(img):
                img[i] = self._normalize(d)
        else:
            img = self._normalize(img)

        return img


class ThresholdIntensity(Transform):
    """Filter the intensity values of whole image to below threshold or above threshold.
    And fill the remaining parts of the image to the `cval` value.

    Args:
        threshold: the threshold to filter intensity values.
        above: filter values above the threshold or below the threshold, default is True.
        cval: value to fill the remaining parts of the image, default is 0.
    """

    def __init__(self, threshold: float, above: bool = True, cval: float = 0.0) -> None:
        threshold = float(threshold)
        assert isinstance(threshold, float), "must set the threshold to filter intensity."
        self.threshold: float = threshold
        self.above: bool = above
        self.cval: float = cval

    def __call__(self, img):
        return np.where(img > self.threshold if self.above else img < self.threshold, img, self.cval).astype(img.dtype)


class ScaleIntensityRange(Transform):
    """Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
    """

    def __init__(self, a_min: float, a_max: float, b_min: float, b_max: float, clip: bool = False) -> None:
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def __call__(self, img):
        img = (img - self.a_min) / (self.a_max - self.a_min)
        img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)

        return img


class AdjustContrast(Transform):
    """Changes image intensity by gamma. Each pixel/voxel intensity is updated as:
        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        gamma: gamma value to adjust the contrast as function.
    """

    def __init__(self, gamma: float) -> None:
        assert isinstance(gamma, float), "gamma must be a float number."
        self.gamma = gamma

    def __call__(self, img):
        epsilon = 1e-7
        img_min = img.min()
        img_range = img.max() - img_min
        return np.power(((img - img_min) / float(img_range + epsilon)), self.gamma) * img_range + img_min


class RandAdjustContrast(Randomizable, Transform):
    """Randomly changes image intensity by gamma. Each pixel/voxel intensity is updated as:
        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        prob: Probability of adjustment.
        gamma (tuple of float or float): Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
    """

    def __init__(self, prob: float = 0.1, gamma=(0.5, 4.5)):
        self.prob = prob
        self.gamma: Tuple[float, float]

        if not isinstance(gamma, (tuple, list)):
            assert gamma > 0.5, "if gamma is single number, must greater than 0.5 and value is picked from (0.5, gamma)"
            self.gamma = (0.5, gamma)
        else:
            assert len(gamma) == 2, "gamma should be a number or pair of numbers."
            self.gamma = (gamma[0], gamma[1])

        self._do_transform = False
        self.gamma_value = None

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random_sample() < self.prob
        self.gamma_value = self.R.uniform(low=self.gamma[0], high=self.gamma[1])

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        adjuster = AdjustContrast(self.gamma_value)
        return adjuster(img)
