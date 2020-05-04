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

import numpy as np

from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.intensity.array import NormalizeIntensity, ScaleIntensityRange, \
    ThresholdIntensity, AdjustContrast, ShiftIntensity, ScaleIntensity


class RandGaussianNoised(Randomizable, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandGaussianNoise`.
    Add Gaussian noise to image. This transform assumes all the expected fields have same shape.

    Args:
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob (float): Probability to add Gaussian noise.
        mean (float or array of floats): Mean or “centre” of the distribution.
        std (float): Standard deviation (spread) of distribution.
    """

    def __init__(self, keys, prob=0.1, mean=0.0, std=0.1):
        super().__init__(keys)
        self.prob = prob
        self.mean = mean
        self.std = std
        self._do_transform = False
        self._noise = None

    def randomize(self, im_shape):
        self._do_transform = self.R.random() < self.prob
        self._noise = self.R.normal(self.mean, self.R.uniform(0, self.std), size=im_shape)

    def __call__(self, data):
        d = dict(data)

        image_shape = d[self.keys[0]].shape  # image shape from the first data key
        self.randomize(image_shape)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = d[key] + self._noise
        return d


class ShiftIntensityd(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.ShiftIntensity`.
    """

    def __init__(self, keys, offset):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offset (int or float): offset value to shift the intensity of image.
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

    def __init__(self, keys, offsets, prob=0.1):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offsets(int, float, tuple or list): offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
        """
        super().__init__(keys)
        self.offsets = (-offsets, offsets) if not isinstance(offsets, (list, tuple)) else offsets
        assert len(self.offsets) == 2, 'offsets should be a number or pair of numbers.'
        self.prob = prob
        self._do_transform = False

    def randomize(self):
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

    def __init__(self, keys, minv=0.0, maxv=1.0, factor=None, dtype=np.float32):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            minv (int or float): minimum value of output data.
            maxv (int or float): maximum value of output data.
            factor (float): factor scale by ``v = v * (1 + factor)``.
            dtype (np.dtype): expected output data type.
        """
        super().__init__(keys)
        self.scaler = ScaleIntensity(minv, maxv, factor, dtype)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.scaler(d[key])
        return d


class RandScaleIntensityd(Randomizable, MapTransform):
    """
    dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    """

    def __init__(self, keys, factors, prob=0.1, dtype=np.float32):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors(float, tuple or list): factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            dtype (np.dtype): expected output data type.
        """
        super().__init__(keys)
        self.factors = (-factors, factors) if not isinstance(factors, (list, tuple)) else factors
        assert len(self.factors) == 2, 'factors should be a number or pair of numbers.'
        self.prob = prob
        self.dtype = dtype
        self._do_transform = False

    def randomize(self):
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        scaler = ScaleIntensity(minv=None, maxv=None, factor=self.factor, dtype=self.dtype)
        for key in self.keys:
            d[key] = scaler(d[key])
        return d


class NormalizeIntensityd(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend (ndarray): the amount to subtract by (usually the mean)
        divisor (ndarray): the amount to divide by (usually the standard deviation)
        nonzero (bool): whether only normalize non-zero values.
        channel_wise (bool): if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(self, keys, subtrahend=None, divisor=None, nonzero=False, channel_wise=False):
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
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        threshold (float or int): the threshold to filter intensity values.
        above (bool): filter values above the threshold or below the threshold, default is True.
        cval (float or int): value to fill the remaining parts of the image, default is 0.
    """

    def __init__(self, keys, threshold, above=True, cval=0):
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
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min (int or float): intensity original range min.
        a_max (int or float): intensity original range max.
        b_min (int or float): intensity target range min.
        b_max (int or float): intensity target range max.
        clip (bool): whether to perform clip after scaling.
    """

    def __init__(self, keys, a_min, a_max, b_min, b_max, clip=False):
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
        gamma (float): gamma value to adjust the contrast as function.
    """

    def __init__(self, keys, gamma):
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
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        prob (float): Probability of adjustment.
        gamma (tuple of float or float): Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
    """

    def __init__(self, keys, prob=0.1, gamma=(0.5, 4.5)):
        super().__init__(keys)
        self.prob = prob
        if not isinstance(gamma, (tuple, list)):
            assert gamma > 0.5, \
                'if gamma is single number, must greater than 0.5 and value is picked from (0.5, gamma)'
            self.gamma = (0.5, gamma)
        else:
            self.gamma = gamma
        assert len(self.gamma) == 2, 'gamma should be a number or pair of numbers.'

        self._do_transform = False
        self.gamma_value = None

    def randomize(self):
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
