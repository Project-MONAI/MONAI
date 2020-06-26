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
A collection of dictionary-based wrappers around the "vanilla" transforms for intensity adjustment
defined in :py:class:`monai.transforms.intensity.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Union, Optional, Tuple

import numpy as np

from monai.config.type_definitions import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.intensity.array import (
    NormalizeIntensity,
    ScaleIntensityRange,
    ThresholdIntensity,
    AdjustContrast,
    ShiftIntensity,
    ScaleIntensity,
)


class RandGaussianNoised(Randomizable, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandGaussianNoise`.
    Add Gaussian noise to image. This transform assumes all the expected fields have same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob: Probability to add Gaussian noise.
        mean (float or array of floats): Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
    """

    def __init__(self, keys: KeysCollection, prob: float = 0.1, mean=0.0, std: float = 0.1):
        super().__init__(keys)
        self.prob = prob
        self.mean = mean
        self.std = std
        self._do_transform = False
        self._noise = None

    def randomize(self, im_shape) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random() < self.prob
        self._noise = self.R.normal(self.mean, self.R.uniform(0, self.std), size=im_shape)

    def __call__(self, data):
        d = dict(data)

        image_shape = d[self.keys[0]].shape  # image shape from the first data key
        self.randomize(image_shape)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = d[key] + self._noise.astype(d[key].dtype)
        return d


class ShiftIntensityd(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.ShiftIntensity`.
    """

    def __init__(self, keys: KeysCollection, offset: float):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offset: offset value to shift the intensity of image.
        """
        super().__init__(keys)
        self.shifter = ShiftIntensity(offset)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.shifter(d[key])
        return d


class RandShiftIntensityd(Randomizable, MapTransform):
    """
    dictionary-based version :py:class:`monai.transforms.RandShiftIntensity`.
    """

    def __init__(self, keys: KeysCollection, offsets, prob: float = 0.1):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offsets(int, float, tuple or list): offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
        """
        super().__init__(keys)
        self.offsets = (-offsets, offsets) if not isinstance(offsets, (list, tuple)) else offsets
        assert len(self.offsets) == 2, "offsets should be a number or pair of numbers."
        self.prob = prob
        self._do_transform = False

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._offset = self.R.uniform(low=self.offsets[0], high=self.offsets[1])
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        shifter = ShiftIntensity(self._offset)
        for key in self.keys:
            d[key] = shifter(d[key])
        return d


class ScaleIntensityd(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensity`.
    Scale the intensity of input image to the given value range (minv, maxv).
    If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
    """

    def __init__(
        self, keys: KeysCollection, minv: float = 0.0, maxv: float = 1.0, factor: Optional[float] = None
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            minv: minimum value of output data.
            maxv: maximum value of output data.
            factor: factor scale by ``v = v * (1 + factor)``.

        """
        super().__init__(keys)
        self.scaler = ScaleIntensity(minv, maxv, factor)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.scaler(d[key])
        return d


class RandScaleIntensityd(Randomizable, MapTransform):
    """
    dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    """

    def __init__(self, keys: KeysCollection, factors, prob: float = 0.1):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors(float, tuple or list): factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)

        """
        super().__init__(keys)
        self.factors = (-factors, factors) if not isinstance(factors, (list, tuple)) else factors
        assert len(self.factors) == 2, "factors should be a number or pair of numbers."
        self.prob = prob
        self._do_transform = False

    def randomize(self) -> None:  # type: ignore # see issue #495
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        scaler = ScaleIntensity(minv=None, maxv=None, factor=self.factor)
        for key in self.keys:
            d[key] = scaler(d[key])
        return d


class NormalizeIntensityd(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend (ndarray): the amount to subtract by (usually the mean)
        divisor (ndarray): the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
        self,
        keys: KeysCollection,
        subtrahend: Optional[np.ndarray] = None,
        divisor: Optional[np.ndarray] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
    ):
        super().__init__(keys)
        self.normalizer = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d


class ThresholdIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ThresholdIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        threshold: the threshold to filter intensity values.
        above: filter values above the threshold or below the threshold, default is True.
        cval: value to fill the remaining parts of the image, default is 0.
    """

    def __init__(self, keys: KeysCollection, threshold: float, above: bool = True, cval: float = 0.0) -> None:
        super().__init__(keys)
        self.filter = ThresholdIntensity(threshold, above, cval)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.filter(d[key])
        return d


class ScaleIntensityRanged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
    """

    def __init__(
        self, keys: KeysCollection, a_min: float, a_max: float, b_min: float, b_max: float, clip: bool = False
    ) -> None:
        super().__init__(keys)
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.scaler(d[key])
        return d


class AdjustContrastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AdjustContrast`.
    Changes image intensity by gamma. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        gamma: gamma value to adjust the contrast as function.
    """

    def __init__(self, keys: KeysCollection, gamma: float):
        super().__init__(keys)
        self.adjuster = AdjustContrast(gamma)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.adjuster(d[key])
        return d


class RandAdjustContrastd(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAdjustContrast`.
    Randomly changes image intensity by gamma. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        prob: Probability of adjustment.
        gamma (tuple of float or float): Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
    """

    def __init__(self, keys: KeysCollection, prob: float = 0.1, gamma: Union[Tuple[float, float], float] = (0.5, 4.5)):
        super().__init__(keys)
        self.prob: float = prob
        self.gamma: Tuple[float, float]

        if not isinstance(gamma, (tuple, list)):
            assert gamma > 0.5, "if gamma is single number, must greater than 0.5 and value is picked from (0.5, gamma)"
            self.gamma = (0.5, gamma)
        else:
            assert len(gamma) == 2, "gamma should be a number or pair of numbers."
            self.gamma = gamma
        assert len(self.gamma) == 2, "gamma should be a number or pair of numbers."

        self._do_transform = False
        self.gamma_value = None

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random_sample() < self.prob
        self.gamma_value = self.R.uniform(low=self.gamma[0], high=self.gamma[1])

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        adjuster = AdjustContrast(self.gamma_value)
        for key in self.keys:
            d[key] = adjuster(d[key])
        return d


RandGaussianNoiseD = RandGaussianNoiseDict = RandGaussianNoised
ShiftIntensityD = ShiftIntensityDict = ShiftIntensityd
RandShiftIntensityD = RandShiftIntensityDict = RandShiftIntensityd
ScaleIntensityD = ScaleIntensityDict = ScaleIntensityd
RandScaleIntensityD = RandScaleIntensityDict = RandScaleIntensityd
NormalizeIntensityD = NormalizeIntensityDict = NormalizeIntensityd
ThresholdIntensityD = ThresholdIntensityDict = ThresholdIntensityd
ScaleIntensityRangeD = ScaleIntensityRangeDict = ScaleIntensityRanged
AdjustContrastD = AdjustContrastDict = AdjustContrastd
RandAdjustContrastD = RandAdjustContrastDict = RandAdjustContrastd
