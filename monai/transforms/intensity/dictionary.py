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
A collection of dictionary-based wrappers around the "vanilla" transforms for intensity adjustment
defined in :py:class:`monai.transforms.intensity.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from collections.abc import Iterable
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.transforms.intensity.array import (
    AdjustContrast,
    GaussianSharpen,
    GaussianSmooth,
    GibbsNoise,
    MaskIntensity,
    NormalizeIntensity,
    RandBiasField,
    RandRicianNoise,
    ScaleIntensity,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    ShiftIntensity,
    StdShiftIntensity,
    ThresholdIntensity,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import dtype_torch_to_numpy, ensure_tuple_rep, ensure_tuple_size

__all__ = [
    "RandGaussianNoised",
    "RandRicianNoised",
    "ShiftIntensityd",
    "RandShiftIntensityd",
    "ScaleIntensityd",
    "RandScaleIntensityd",
    "StdShiftIntensityd",
    "RandStdShiftIntensityd",
    "RandBiasFieldd",
    "NormalizeIntensityd",
    "ThresholdIntensityd",
    "ScaleIntensityRanged",
    "AdjustContrastd",
    "RandAdjustContrastd",
    "ScaleIntensityRangePercentilesd",
    "MaskIntensityd",
    "GaussianSmoothd",
    "RandGaussianSmoothd",
    "GaussianSharpend",
    "RandGaussianSharpend",
    "GibbsNoised",
    "RandGibbsNoised",
    "RandHistogramShiftd",
    "RandGaussianNoiseD",
    "RandGaussianNoiseDict",
    "ShiftIntensityD",
    "ShiftIntensityDict",
    "RandShiftIntensityD",
    "RandShiftIntensityDict",
    "ScaleIntensityD",
    "ScaleIntensityDict",
    "StdShiftIntensityD",
    "StdShiftIntensityDict",
    "RandScaleIntensityD",
    "RandScaleIntensityDict",
    "RandStdShiftIntensityD",
    "RandStdShiftIntensityDict",
    "RandBiasFieldD",
    "RandBiasFieldDict",
    "NormalizeIntensityD",
    "NormalizeIntensityDict",
    "ThresholdIntensityD",
    "ThresholdIntensityDict",
    "ScaleIntensityRangeD",
    "ScaleIntensityRangeDict",
    "AdjustContrastD",
    "AdjustContrastDict",
    "RandAdjustContrastD",
    "RandAdjustContrastDict",
    "ScaleIntensityRangePercentilesD",
    "ScaleIntensityRangePercentilesDict",
    "MaskIntensityD",
    "MaskIntensityDict",
    "GaussianSmoothD",
    "GaussianSmoothDict",
    "RandGaussianSmoothD",
    "RandGaussianSmoothDict",
    "GaussianSharpenD",
    "GaussianSharpenDict",
    "RandGaussianSharpenD",
    "RandGaussianSharpenDict",
    "GibbsNoiseD",
    "GibbsNoiseDict",
    "RandGibbsNoiseD",
    "RandGibbsNoiseDict",
    "RandHistogramShiftD",
    "RandHistogramShiftDict",
    "RandRicianNoiseD",
    "RandRicianNoiseDict",
]


class RandGaussianNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandGaussianNoise`.
    Add Gaussian noise to image. This transform assumes all the expected fields have same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        mean: Union[Sequence[float], float] = 0.0,
        std: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.mean = ensure_tuple_rep(mean, len(self.keys))
        self.std = std
        self._noise: List[np.ndarray] = []

    def randomize(self, im_shape: Sequence[int]) -> None:
        super().randomize(None)
        self._noise.clear()
        for m in self.mean:
            self._noise.append(self.R.normal(m, self.R.uniform(0, self.std), size=im_shape))

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        image_shape = d[self.keys[0]].shape  # image shape from the first data key
        self.randomize(image_shape)
        if len(self._noise) != len(self.keys):
            raise AssertionError
        if not self._do_transform:
            return d
        for key, noise in self.key_iterator(d, self._noise):
            dtype = dtype_torch_to_numpy(d[key].dtype) if isinstance(d[key], torch.Tensor) else d[key].dtype
            d[key] = d[key] + noise.astype(dtype)
        return d


class RandRicianNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRicianNoise`.
    Add Rician noise to image. This transform assumes all the expected fields have same shape.

    Args:
        keys: Keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        global_prob: Probability to add Rician noise to the dictionary.
        prob: Probability to add Rician noise to each item in the dictionary,
            once asserted that noise will be added to the dictionary at all.
        mean: Mean or "centre" of the Gaussian distributions sampled to make up
            the Rician noise.
        std: Standard deviation (spread) of the Gaussian distributions sampled
            to make up the Rician noise.
        channel_wise: If True, treats each channel of the image separately.
        relative: If True, the spread of the sampled Gaussian distributions will
            be std times the standard deviation of the image or channel's intensity
            histogram.
        sample_std: If True, sample the spread of the Gaussian distributions
            uniformly from 0 to std.
        allow_missing_keys: Don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        global_prob: float = 0.1,
        prob: float = 1.0,
        mean: Union[Sequence[float], float] = 0.0,
        std: Union[Sequence[float], float] = 1.0,
        channel_wise: bool = False,
        relative: bool = False,
        sample_std: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, global_prob)
        self.rand_rician_noise = RandRicianNoise(prob, mean, std, channel_wise, relative, sample_std)

    def __call__(
        self, data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[Hashable, Union[torch.Tensor, np.ndarray]]:
        d = dict(data)
        super().randomize(None)
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            d[key] = self.rand_rician_noise(d[key])
        return d


class ShiftIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ShiftIntensity`.
    """

    def __init__(self, keys: KeysCollection, offset: float, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offset: offset value to shift the intensity of image.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.shifter = ShiftIntensity(offset)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.shifter(d[key])
        return d


class RandShiftIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandShiftIntensity`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        offsets: Union[Tuple[float, float], float],
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offsets: offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        if isinstance(offsets, (int, float)):
            self.offsets = (min(-offsets, offsets), max(-offsets, offsets))
        else:
            if len(offsets) != 2:
                raise AssertionError("offsets should be a number or pair of numbers.")
            self.offsets = (min(offsets), max(offsets))
        self._offset = self.offsets[0]

    def randomize(self, data: Optional[Any] = None) -> None:
        self._offset = self.R.uniform(low=self.offsets[0], high=self.offsets[1])
        super().randomize(None)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        shifter = ShiftIntensity(self._offset)
        for key in self.key_iterator(d):
            d[key] = shifter(d[key])
        return d


class StdShiftIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.StdShiftIntensity`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        factor: float,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factor: factor shift by ``v = v + factor * std(v)``.
            nonzero: whether only count non-zero values.
            channel_wise: if True, calculate on each channel separately. Please ensure
                that the first dimension represents the channel of the image if True.
            dtype: output data type, defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.shifter = StdShiftIntensity(factor, nonzero, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.shifter(d[key])
        return d


class RandStdShiftIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandStdShiftIntensity`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        factors: Union[Tuple[float, float], float],
        prob: float = 0.1,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors: if tuple, the randomly picked range is (min(factors), max(factors)).
                If single number, the range is (-factors, factors).
            prob: probability of std shift.
            nonzero: whether only count non-zero values.
            channel_wise: if True, calculate on each channel separately.
            dtype: output data type, defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        if isinstance(factors, (int, float)):
            self.factors = (min(-factors, factors), max(-factors, factors))
        else:
            if len(factors) != 2:
                raise AssertionError("factors should be a number or pair of numbers.")
            self.factors = (min(factors), max(factors))
        self.factor = self.factors[0]
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    def randomize(self, data: Optional[Any] = None) -> None:
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])
        super().randomize(None)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        shifter = StdShiftIntensity(self.factor, self.nonzero, self.channel_wise, self.dtype)
        for key in self.key_iterator(d):
            d[key] = shifter(d[key])
        return d


class ScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensity`.
    Scale the intensity of input image to the given value range (minv, maxv).
    If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
    """

    def __init__(
        self,
        keys: KeysCollection,
        minv: Optional[float] = 0.0,
        maxv: Optional[float] = 1.0,
        factor: Optional[float] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            minv: minimum value of output data.
            maxv: maximum value of output data.
            factor: factor scale by ``v = v * (1 + factor)``. In order to use
                this parameter, please set `minv` and `maxv` into None.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensity(minv, maxv, factor)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class RandScaleIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        factors: Union[Tuple[float, float], float],
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        if isinstance(factors, (int, float)):
            self.factors = (min(-factors, factors), max(-factors, factors))
        else:
            if len(factors) != 2:
                raise AssertionError("factors should be a number or pair of numbers.")
            self.factors = (min(factors), max(factors))
        self.factor = self.factors[0]

    def randomize(self, data: Optional[Any] = None) -> None:
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])
        super().randomize(None)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        scaler = ScaleIntensity(minv=None, maxv=None, factor=self.factor)
        for key in self.key_iterator(d):
            d[key] = scaler(d[key])
        return d


class RandBiasFieldd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandBiasField`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        degree: int = 3,
        coeff_range: Tuple[float, float] = (0.0, 0.1),
        dtype: DtypeLike = np.float32,
        prob: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            degree: degree of freedom of the polynomials. The value should be no less than 1.
                Defaults to 3.
            coeff_range: range of the random coefficients. Defaults to (0.0, 0.1).
            dtype: output data type, defaults to float32.
            prob: probability to do random bias field.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.rand_bias_field = RandBiasField(degree, coeff_range, dtype, prob)

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            d[key] = self.rand_bias_field(d[key])
        return d


class NormalizeIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
        dtype: output data type, defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        subtrahend: Optional[np.ndarray] = None,
        divisor: Optional[np.ndarray] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
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
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        threshold: float,
        above: bool = True,
        cval: float = 0.0,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.filter = ThresholdIntensity(threshold, above, cval)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
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
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        b_min: float,
        b_max: float,
        clip: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class AdjustContrastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AdjustContrast`.
    Changes image intensity by gamma. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        gamma: gamma value to adjust the contrast as function.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(self, keys: KeysCollection, gamma: float, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.adjuster = AdjustContrast(gamma)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adjuster(d[key])
        return d


class RandAdjustContrastd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAdjustContrast`.
    Randomly changes image intensity by gamma. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        gamma: Union[Tuple[float, float], float] = (0.5, 4.5),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        if isinstance(gamma, (int, float)):
            if gamma <= 0.5:
                raise AssertionError(
                    "if gamma is single number, must greater than 0.5 and value is picked from (0.5, gamma)"
                )
            self.gamma = (0.5, gamma)
        else:
            if len(gamma) != 2:
                raise AssertionError("gamma should be a number or pair of numbers.")
            self.gamma = (min(gamma), max(gamma))

        self.gamma_value: Optional[float] = None

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.gamma_value = self.R.uniform(low=self.gamma[0], high=self.gamma[1])

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if self.gamma_value is None:
            raise AssertionError
        if not self._do_transform:
            return d
        adjuster = AdjustContrast(self.gamma_value)
        for key in self.key_iterator(d):
            d[key] = adjuster(d[key])
        return d


class ScaleIntensityRangePercentilesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRangePercentiles`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: lower percentile.
        upper: upper percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max]
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        lower: float,
        upper: float,
        b_min: float,
        b_max: float,
        clip: bool = False,
        relative: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRangePercentiles(lower, upper, b_min, b_max, clip, relative)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class MaskIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MaskIntensity`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        mask_data: if mask data is single channel, apply to every channel
            of input image. if multiple channels, the channel number must
            match input data. mask_data will be converted to `bool` values
            by `mask_data > 0` before applying transform to input image.
            if None, will extract the mask data from input data based on `mask_key`.
        mask_key: the key to extract mask data from input dictionary, only works
            when `mask_data` is None.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_data: Optional[np.ndarray] = None,
        mask_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = MaskIntensity(mask_data)
        self.mask_key = mask_key if mask_data is None else None

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], d[self.mask_key]) if self.mask_key is not None else self.converter(d[key])
        return d


class GaussianSmoothd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: Union[Sequence[float], float],
        approx: str = "erf",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GaussianSmooth(sigma, approx=approx)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class RandGaussianSmoothd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma_x: randomly select sigma value for the first spatial dimension.
        sigma_y: randomly select sigma value for the second spatial dimension if have.
        sigma_z: randomly select sigma value for the third spatial dimension if have.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        prob: probability of Gaussian smooth.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma_x: Tuple[float, float] = (0.25, 1.5),
        sigma_y: Tuple[float, float] = (0.25, 1.5),
        sigma_z: Tuple[float, float] = (0.25, 1.5),
        approx: str = "erf",
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.sigma_x, self.sigma_y, self.sigma_z = sigma_x, sigma_y, sigma_z
        self.approx = approx

        self.x, self.y, self.z = self.sigma_x[0], self.sigma_y[0], self.sigma_z[0]

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.x = self.R.uniform(low=self.sigma_x[0], high=self.sigma_x[1])
        self.y = self.R.uniform(low=self.sigma_y[0], high=self.sigma_y[1])
        self.z = self.R.uniform(low=self.sigma_z[0], high=self.sigma_z[1])

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            sigma = ensure_tuple_size(tup=(self.x, self.y, self.z), dim=d[key].ndim - 1)
            d[key] = GaussianSmooth(sigma=sigma, approx=self.approx)(d[key])
        return d


class GaussianSharpend(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSharpen`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma1: sigma parameter for the first gaussian kernel. if a list of values, must match the count
            of spatial dimensions of input data, and apply every value in the list to 1 spatial dimension.
            if only 1 value provided, use it for all spatial dimensions.
        sigma2: sigma parameter for the second gaussian kernel. if a list of values, must match the count
            of spatial dimensions of input data, and apply every value in the list to 1 spatial dimension.
            if only 1 value provided, use it for all spatial dimensions.
        alpha: weight parameter to compute the final result.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma1: Union[Sequence[float], float] = 3.0,
        sigma2: Union[Sequence[float], float] = 1.0,
        alpha: float = 30.0,
        approx: str = "erf",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GaussianSharpen(sigma1, sigma2, alpha, approx=approx)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class RandGaussianSharpend(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSharpen`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma1_x: randomly select sigma value for the first spatial dimension of first gaussian kernel.
        sigma1_y: randomly select sigma value for the second spatial dimension(if have) of first gaussian kernel.
        sigma1_z: randomly select sigma value for the third spatial dimension(if have) of first gaussian kernel.
        sigma2_x: randomly select sigma value for the first spatial dimension of second gaussian kernel.
            if only 1 value `X` provided, it must be smaller than `sigma1_x` and randomly select from [X, sigma1_x].
        sigma2_y: randomly select sigma value for the second spatial dimension(if have) of second gaussian kernel.
            if only 1 value `Y` provided, it must be smaller than `sigma1_y` and randomly select from [Y, sigma1_y].
        sigma2_z: randomly select sigma value for the third spatial dimension(if have) of second gaussian kernel.
            if only 1 value `Z` provided, it must be smaller than `sigma1_z` and randomly select from [Z, sigma1_z].
        alpha: randomly select weight parameter to compute the final result.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        prob: probability of Gaussian sharpen.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma1_x: Tuple[float, float] = (0.5, 1.0),
        sigma1_y: Tuple[float, float] = (0.5, 1.0),
        sigma1_z: Tuple[float, float] = (0.5, 1.0),
        sigma2_x: Union[Tuple[float, float], float] = 0.5,
        sigma2_y: Union[Tuple[float, float], float] = 0.5,
        sigma2_z: Union[Tuple[float, float], float] = 0.5,
        alpha: Tuple[float, float] = (10.0, 30.0),
        approx: str = "erf",
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.sigma1_x = sigma1_x
        self.sigma1_y = sigma1_y
        self.sigma1_z = sigma1_z
        self.sigma2_x = sigma2_x
        self.sigma2_y = sigma2_y
        self.sigma2_z = sigma2_z
        self.alpha = alpha
        self.approx = approx

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.x1 = self.R.uniform(low=self.sigma1_x[0], high=self.sigma1_x[1])
        self.y1 = self.R.uniform(low=self.sigma1_y[0], high=self.sigma1_y[1])
        self.z1 = self.R.uniform(low=self.sigma1_z[0], high=self.sigma1_z[1])
        sigma2_x = (self.sigma2_x, self.x1) if not isinstance(self.sigma2_x, Iterable) else self.sigma2_x
        sigma2_y = (self.sigma2_y, self.y1) if not isinstance(self.sigma2_y, Iterable) else self.sigma2_y
        sigma2_z = (self.sigma2_z, self.z1) if not isinstance(self.sigma2_z, Iterable) else self.sigma2_z
        self.x2 = self.R.uniform(low=sigma2_x[0], high=sigma2_x[1])
        self.y2 = self.R.uniform(low=sigma2_y[0], high=sigma2_y[1])
        self.z2 = self.R.uniform(low=sigma2_z[0], high=sigma2_z[1])
        self.a = self.R.uniform(low=self.alpha[0], high=self.alpha[1])

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            sigma1 = ensure_tuple_size(tup=(self.x1, self.y1, self.z1), dim=d[key].ndim - 1)
            sigma2 = ensure_tuple_size(tup=(self.x2, self.y2, self.z2), dim=d[key].ndim - 1)
            d[key] = GaussianSharpen(sigma1=sigma1, sigma2=sigma2, alpha=self.a, approx=self.approx)(d[key])
        return d


class RandHistogramShiftd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandHistogramShift`.
    Apply random nonlinear transform the the image's intensity histogram.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        num_control_points: number of control points governing the nonlinear intensity mapping.
            a smaller number of control points allows for larger intensity shifts. if two values provided, number of
            control points selecting from range (min_value, max_value).
        prob: probability of histogram shift.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        num_control_points: Union[Tuple[int, int], int] = 10,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        if isinstance(num_control_points, int):
            if num_control_points <= 2:
                raise AssertionError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (num_control_points, num_control_points)
        else:
            if len(num_control_points) != 2:
                raise AssertionError("num_control points should be a number or a pair of numbers")
            if min(num_control_points) <= 2:
                raise AssertionError("num_control_points should be greater than or equal to 3")
            self.num_control_points = (min(num_control_points), max(num_control_points))

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        num_control_point = self.R.randint(self.num_control_points[0], self.num_control_points[1] + 1)
        self.reference_control_points = np.linspace(0, 1, num_control_point)
        self.floating_control_points = np.copy(self.reference_control_points)
        for i in range(1, num_control_point - 1):
            self.floating_control_points[i] = self.R.uniform(
                self.floating_control_points[i - 1], self.floating_control_points[i + 1]
            )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            img_min, img_max = d[key].min(), d[key].max()
            reference_control_points_scaled = self.reference_control_points * (img_max - img_min) + img_min
            floating_control_points_scaled = self.floating_control_points * (img_max - img_min) + img_min
            dtype = d[key].dtype
            d[key] = np.interp(d[key], reference_control_points_scaled, floating_control_points_scaled).astype(dtype)
        return d


class RandGibbsNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based version of RandGibbsNoise.

    Naturalistic image augmentation via Gibbs artifacts. The transform
    randomly applies Gibbs noise to 2D/3D MRI images. Gibbs artifacts
    are one of the common type of type artifacts appearing in MRI scans.

    The transform is applied to all the channels in the data.

    For general information on Gibbs artifacts, please refer to:
    https://pubs.rsna.org/doi/full/10.1148/rg.313105115
    https://pubs.rsna.org/doi/full/10.1148/radiographics.22.4.g02jl14949

    Args:
        keys: 'image', 'label', or ['image', 'label'] depending on which data
                you need to transform.
        prob (float): probability of applying the transform.
        alpha (float, List[float]): Parametrizes the intensity of the Gibbs noise filter applied. Takes
            values in the interval [0,1] with alpha = 0 acting as the identity mapping.
            If a length-2 list is given as [a,b] then the value of alpha will be sampled
            uniformly from the interval [a,b].
        as_tensor_output: if true return torch.Tensor, else return np.array. default: True.
        allow_missing_keys: do not raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        alpha: Sequence[float] = (0.0, 1.0),
        as_tensor_output: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.alpha = alpha
        self.sampled_alpha = -1.0  # stores last alpha sampled by randomize()
        self.as_tensor_output = as_tensor_output

    def __call__(
        self, data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[Hashable, Union[torch.Tensor, np.ndarray]]:

        d = dict(data)
        self._randomize(None)

        for i, key in enumerate(self.key_iterator(d)):
            if self._do_transform:
                if i == 0:
                    transform = GibbsNoise(self.sampled_alpha, self.as_tensor_output)
                d[key] = transform(d[key])
            else:
                if isinstance(d[key], np.ndarray) and self.as_tensor_output:
                    d[key] = torch.Tensor(d[key])
                elif isinstance(d[key], torch.Tensor) and not self.as_tensor_output:
                    d[key] = self._to_numpy(d[key])
        return d

    def _randomize(self, _: Any) -> None:
        """
        (1) Set random variable to apply the transform.
        (2) Get alpha from uniform distribution.
        """
        super().randomize(None)
        self.sampled_alpha = self.R.uniform(self.alpha[0], self.alpha[1])

    def _to_numpy(self, d: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(d, torch.Tensor):
            d_numpy: np.ndarray = d.cpu().detach().numpy()
        return d_numpy


class GibbsNoised(MapTransform):
    """
    Dictionary-based version of GibbsNoise.

    The transform applies Gibbs noise to 2D/3D MRI images. Gibbs artifacts
    are one of the common type of type artifacts appearing in MRI scans.

    For general information on Gibbs artifacts, please refer to:
    https://pubs.rsna.org/doi/full/10.1148/rg.313105115
    https://pubs.rsna.org/doi/full/10.1148/radiographics.22.4.g02jl14949

    Args:
        keys: 'image', 'label', or ['image', 'label'] depending on which data
                you need to transform.
        alpha (float): Parametrizes the intensity of the Gibbs noise filter applied. Takes
            values in the interval [0,1] with alpha = 0 acting as the identity mapping.
        as_tensor_output: if true return torch.Tensor, else return np.array. default: True.
        allow_missing_keys: do not raise exception if key is missing.
    """

    def __init__(
        self, keys: KeysCollection, alpha: float = 0.5, as_tensor_output: bool = True, allow_missing_keys: bool = False
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.transform = GibbsNoise(alpha, as_tensor_output)

    def __call__(
        self, data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]]
    ) -> Dict[Hashable, Union[torch.Tensor, np.ndarray]]:

        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


RandGaussianNoiseD = RandGaussianNoiseDict = RandGaussianNoised
RandRicianNoiseD = RandRicianNoiseDict = RandRicianNoised
ShiftIntensityD = ShiftIntensityDict = ShiftIntensityd
RandShiftIntensityD = RandShiftIntensityDict = RandShiftIntensityd
StdShiftIntensityD = StdShiftIntensityDict = StdShiftIntensityd
RandStdShiftIntensityD = RandStdShiftIntensityDict = RandStdShiftIntensityd
RandBiasFieldD = RandBiasFieldDict = RandBiasFieldd
ScaleIntensityD = ScaleIntensityDict = ScaleIntensityd
RandScaleIntensityD = RandScaleIntensityDict = RandScaleIntensityd
NormalizeIntensityD = NormalizeIntensityDict = NormalizeIntensityd
ThresholdIntensityD = ThresholdIntensityDict = ThresholdIntensityd
ScaleIntensityRangeD = ScaleIntensityRangeDict = ScaleIntensityRanged
AdjustContrastD = AdjustContrastDict = AdjustContrastd
RandAdjustContrastD = RandAdjustContrastDict = RandAdjustContrastd
ScaleIntensityRangePercentilesD = ScaleIntensityRangePercentilesDict = ScaleIntensityRangePercentilesd
MaskIntensityD = MaskIntensityDict = MaskIntensityd
GaussianSmoothD = GaussianSmoothDict = GaussianSmoothd
RandGaussianSmoothD = RandGaussianSmoothDict = RandGaussianSmoothd
GaussianSharpenD = GaussianSharpenDict = GaussianSharpend
RandGaussianSharpenD = RandGaussianSharpenDict = RandGaussianSharpend
RandHistogramShiftD = RandHistogramShiftDict = RandHistogramShiftd
RandGibbsNoiseD = RandGibbsNoiseDict = RandGibbsNoised
GibbsNoiseD = GibbsNoiseDict = GibbsNoised
