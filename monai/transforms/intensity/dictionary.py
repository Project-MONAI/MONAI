# Copyright (c) MONAI Consortium
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

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Callable

import numpy as np

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.intensity.array import (
    AdjustContrast,
    ClipIntensityPercentiles,
    ComputeHoVerMaps,
    ForegroundMask,
    GaussianSharpen,
    GaussianSmooth,
    GibbsNoise,
    HistogramNormalize,
    KSpaceSpikeNoise,
    MaskIntensity,
    MedianSmooth,
    NormalizeIntensity,
    RandAdjustContrast,
    RandBiasField,
    RandCoarseDropout,
    RandCoarseShuffle,
    RandGaussianNoise,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGibbsNoise,
    RandHistogramShift,
    RandKSpaceSpikeNoise,
    RandRicianNoise,
    RandScaleIntensity,
    RandScaleIntensityFixedMean,
    RandShiftIntensity,
    RandStdShiftIntensity,
    SavitzkyGolaySmooth,
    ScaleIntensity,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    ShiftIntensity,
    StdShiftIntensity,
    ThresholdIntensity,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utils import is_positive
from monai.utils import convert_to_tensor, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix

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
    "ClipIntensityPercentilesd",
    "AdjustContrastd",
    "RandAdjustContrastd",
    "ScaleIntensityRangePercentilesd",
    "MaskIntensityd",
    "SavitzkyGolaySmoothd",
    "MedianSmoothd",
    "GaussianSmoothd",
    "RandGaussianSmoothd",
    "GaussianSharpend",
    "RandGaussianSharpend",
    "GibbsNoised",
    "RandGibbsNoised",
    "KSpaceSpikeNoised",
    "RandKSpaceSpikeNoised",
    "RandHistogramShiftd",
    "RandCoarseDropoutd",
    "RandCoarseShuffled",
    "HistogramNormalized",
    "ForegroundMaskd",
    "ComputeHoVerMapsd",
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
    "RandScaleIntensityFixedMeand",
    "RandScaleIntensityFixedMeanDict",
    "RandScaleIntensityFixedMeanD",
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
    "ClipIntensityPercentilesD",
    "ClipIntensityPercentilesDict",
    "AdjustContrastD",
    "AdjustContrastDict",
    "RandAdjustContrastD",
    "RandAdjustContrastDict",
    "ScaleIntensityRangePercentilesD",
    "ScaleIntensityRangePercentilesDict",
    "MaskIntensityD",
    "MaskIntensityDict",
    "SavitzkyGolaySmoothD",
    "SavitzkyGolaySmoothDict",
    "MedianSmoothD",
    "MedianSmoothDict",
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
    "KSpaceSpikeNoiseD",
    "KSpaceSpikeNoiseDict",
    "RandHistogramShiftD",
    "RandHistogramShiftDict",
    "RandRicianNoiseD",
    "RandRicianNoiseDict",
    "RandCoarseDropoutD",
    "RandCoarseDropoutDict",
    "RandCoarseShuffleD",
    "RandCoarseShuffleDict",
    "HistogramNormalizeD",
    "HistogramNormalizeDict",
    "RandKSpaceSpikeNoiseD",
    "RandKSpaceSpikeNoiseDict",
    "ForegroundMaskD",
    "ForegroundMaskDict",
    "ComputeHoVerMapsD",
    "ComputeHoVerMapsDict",
]

DEFAULT_POST_FIX = PostFix.meta()


class RandGaussianNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandGaussianNoise`.
    Add Gaussian noise to image. This transform assumes all the expected fields have same shape, if you want to add
    different noise for every field, please use this transform separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
        sample_std: If True, sample the spread of the Gaussian distribution uniformly from 0 to std.
    """

    backend = RandGaussianNoise.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        mean: float = 0.0,
        std: float = 0.1,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
        sample_std: bool = True,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_gaussian_noise = RandGaussianNoise(mean=mean, std=std, prob=1.0, dtype=dtype, sample_std=sample_std)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandGaussianNoised:
        super().set_random_state(seed, state)
        self.rand_gaussian_noise.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random noise
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_gaussian_noise.randomize(d[first_key])

        for key in self.key_iterator(d):
            d[key] = self.rand_gaussian_noise(img=d[key], randomize=False)
        return d


class RandRicianNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRicianNoise`.
    Add Rician noise to image. This transform assumes all the expected fields have same shape, if want to add
    different noise for every field, please use this transform separately.

    Args:
        keys: Keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob: Probability to add Rician noise to the dictionary.
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
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: Don't raise exception if key is missing.
    """

    backend = RandRicianNoise.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        mean: Sequence[float] | float = 0.0,
        std: Sequence[float] | float = 1.0,
        channel_wise: bool = False,
        relative: bool = False,
        sample_std: bool = True,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_rician_noise = RandRicianNoise(
            prob=1.0,
            mean=mean,
            std=std,
            channel_wise=channel_wise,
            relative=relative,
            sample_std=sample_std,
            dtype=dtype,
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandRicianNoised:
        super().set_random_state(seed, state)
        self.rand_rician_noise.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        for key in self.key_iterator(d):
            d[key] = self.rand_rician_noise(d[key], randomize=True)
        return d


class ShiftIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ShiftIntensity`.
    """

    backend = ShiftIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        offset: float,
        safe: bool = False,
        factor_key: str | None = None,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offset: offset value to shift the intensity of image.
            safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
                E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
            factor_key: if not None, use it as the key to extract a value from the corresponding
                metadata dictionary of `key` at runtime, and multiply the `offset` to shift intensity.
                Usually, `IntensityStatsd` transform can pre-compute statistics of intensity values
                and store in the metadata.
                it also can be a sequence of strings, map to `keys`.
            meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
                used to extract the factor value is `factor_key` is not None.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
                to the key data, default is `meta_dict`, the metadata is a dictionary object.
                used to extract the factor value is `factor_key` is not None.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.factor_key = ensure_tuple_rep(factor_key, len(self.keys))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.shifter = ShiftIntensity(offset, safe)

    def __call__(self, data) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, factor_key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.factor_key, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            factor: float | None = d[meta_key].get(factor_key) if meta_key in d else None
            offset = None if factor is None else self.shifter.offset * factor
            d[key] = self.shifter(d[key], offset=offset)
        return d


class RandShiftIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandShiftIntensity`.
    """

    backend = RandShiftIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        offsets: tuple[float, float] | float,
        safe: bool = False,
        factor_key: str | None = None,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        prob: float = 0.1,
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            offsets: offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
                E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
            factor_key: if not None, use it as the key to extract a value from the corresponding
                metadata dictionary of `key` at runtime, and multiply the random `offset` to shift intensity.
                Usually, `IntensityStatsd` transform can pre-compute statistics of intensity values
                and store in the metadata.
                it also can be a sequence of strings, map to `keys`.
            meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
                used to extract the factor value is `factor_key` is not None.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to fetch the metadata according
                to the key data, default is `meta_dict`, the metadata is a dictionary object.
                used to extract the factor value is `factor_key` is not None.
            prob: probability of shift.
                (Default 0.1, with 10% probability it returns an array shifted intensity.)
            channel_wise: if True, shift intensity on each channel separately. For each channel, a random offset will be chosen.
                Please ensure that the first dimension represents the channel of the image if True.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.factor_key = ensure_tuple_rep(factor_key, len(self.keys))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.shifter = RandShiftIntensity(offsets=offsets, safe=safe, prob=1.0, channel_wise=channel_wise)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandShiftIntensityd:
        super().set_random_state(seed, state)
        self.shifter.set_random_state(seed, state)
        return self

    def __call__(self, data) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random shift factor
        self.shifter.randomize(d[first_key])
        for key, factor_key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.factor_key, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            factor: float | None = d[meta_key].get(factor_key) if meta_key in d else None
            d[key] = self.shifter(d[key], factor=factor, randomize=False)
        return d


class StdShiftIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.StdShiftIntensity`.
    """

    backend = StdShiftIntensity.backend

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
            dtype: output data type, if None, same as input image. defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.shifter = StdShiftIntensity(factor, nonzero, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.shifter(d[key])
        return d


class RandStdShiftIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandStdShiftIntensity`.
    """

    backend = RandStdShiftIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        factors: tuple[float, float] | float,
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
            dtype: output data type, if None, same as input image. defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.shifter = RandStdShiftIntensity(
            factors=factors, nonzero=nonzero, channel_wise=channel_wise, dtype=dtype, prob=1.0
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandStdShiftIntensityd:
        super().set_random_state(seed, state)
        self.shifter.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random shift factor
        self.shifter.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.shifter(d[key], randomize=False)
        return d


class ScaleIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensity`.
    Scale the intensity of input image to the given value range (minv, maxv).
    If `minv` and `maxv` not provided, use `factor` to scale image by ``v = v * (1 + factor)``.
    """

    backend = ScaleIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        minv: float | None = 0.0,
        maxv: float | None = 1.0,
        factor: float | None = None,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            minv: minimum value of output data.
            maxv: maximum value of output data.
            factor: factor scale by ``v = v * (1 + factor)``. In order to use
                this parameter, please set both `minv` and `maxv` into None.
            channel_wise: if True, scale on each channel separately. Please ensure
                that the first dimension represents the channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensity(minv, maxv, factor, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class RandScaleIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    """

    backend = RandScaleIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        factors: tuple[float, float] | float,
        prob: float = 0.1,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of scale.
                (Default 0.1, with 10% probability it returns a scaled array.)
            channel_wise: if True, scale on each channel separately. Please ensure
                that the first dimension represents the channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.scaler = RandScaleIntensity(factors=factors, dtype=dtype, prob=1.0, channel_wise=channel_wise)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandScaleIntensityd:
        super().set_random_state(seed, state)
        self.scaler.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random scale factor
        self.scaler.randomize(d[first_key])
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key], randomize=False)
        return d


class RandScaleIntensityFixedMeand(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleIntensity`.
    Subtract the mean intensity before scaling with `factor`, then add the same value after scaling
    to ensure that the output has the same mean as the input.
    """

    backend = RandScaleIntensityFixedMean.backend

    def __init__(
        self,
        keys: KeysCollection,
        factors: Sequence[float] | float,
        fixed_mean: bool = True,
        preserve_range: bool = False,
        prob: float = 0.1,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            preserve_range: clips the output array/tensor to the range of the input array/tensor
            fixed_mean: subtract the mean intensity before scaling with `factor`, then add the same value after scaling
                to ensure that the output has the same mean as the input.
            channel_wise: if True, scale on each channel separately. `preserve_range` and `fixed_mean` are also applied
            on each channel separately if `channel_wise` is True. Please ensure that the first dimension represents the
            channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.fixed_mean = fixed_mean
        self.preserve_range = preserve_range
        self.scaler = RandScaleIntensityFixedMean(
            factors=factors, fixed_mean=self.fixed_mean, preserve_range=preserve_range, dtype=dtype, prob=1.0
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandScaleIntensityFixedMeand:
        super().set_random_state(seed, state)
        self.scaler.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random scale factor
        self.scaler.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key], randomize=False)
        return d


class RandBiasFieldd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandBiasField`.
    """

    backend = RandBiasField.backend

    def __init__(
        self,
        keys: KeysCollection,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
        dtype: DtypeLike = np.float32,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            degree: degree of freedom of the polynomials. The value should be no less than 1.
                Defaults to 3.
            coeff_range: range of the random coefficients. Defaults to (0.0, 0.1).
            dtype: output data type, if None, same as input image. defaults to float32.
            prob: probability to do random bias field.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.rand_bias_field = RandBiasField(degree=degree, coeff_range=coeff_range, dtype=dtype, prob=1.0)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandBiasFieldd:
        super().set_random_state(seed, state)
        self.rand_bias_field.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random bias factor
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_bias_field.randomize(img_size=d[first_key].shape[1:])

        for key in self.key_iterator(d):
            d[key] = self.rand_bias_field(d[key], randomize=False)
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
        channel_wise: if True, calculate on each channel separately, otherwise, calculate on
            the entire image directly. default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = NormalizeIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        subtrahend: NdarrayOrTensor | None = None,
        divisor: NdarrayOrTensor | None = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
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

    backend = ThresholdIntensity.backend

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

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
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
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        b_min: float | None = None,
        b_max: float | None = None,
        clip: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class ClipIntensityPercentilesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ClipIntensityPercentiles`.
    Clip the intensity values of input image to a specific range based on the intensity distribution of the input.
    If `sharpness_factor` is provided, the intensity values will be soft clipped according to
    f(x) = x + (1/sharpness_factor) * softplus(- c(x - minv)) - (1/sharpness_factor)*softplus(c(x - maxv))
    """

    def __init__(
        self,
        keys: KeysCollection,
        lower: float | None,
        upper: float | None,
        sharpness_factor: float | None = None,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ClipIntensityPercentiles(
            lower=lower, upper=upper, sharpness_factor=sharpness_factor, channel_wise=channel_wise, dtype=dtype
        )

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d


class AdjustContrastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AdjustContrast`.
    Changes image intensity with gamma transform. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        gamma: gamma value to adjust the contrast as function.
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = AdjustContrast.backend

    def __init__(
        self,
        keys: KeysCollection,
        gamma: float,
        invert_image: bool = False,
        retain_stats: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.adjuster = AdjustContrast(gamma, invert_image, retain_stats)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adjuster(d[key])
        return d


class RandAdjustContrastd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAdjustContrast`.
    Randomly changes image intensity with gamma transform. Each pixel/voxel intensity is updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandAdjustContrast.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        gamma: tuple[float, float] | float = (0.5, 4.5),
        invert_image: bool = False,
        retain_stats: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.adjuster = RandAdjustContrast(gamma=gamma, prob=1.0, invert_image=invert_image, retain_stats=retain_stats)
        self.invert_image = invert_image

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandAdjustContrastd:
        super().set_random_state(seed, state)
        self.adjuster.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random gamma value
        self.adjuster.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.adjuster(d[key], randomize=False)
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
        channel_wise: if True, compute intensity percentile and normalize every channel separately.
            default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRangePercentiles.backend

    def __init__(
        self,
        keys: KeysCollection,
        lower: float,
        upper: float,
        b_min: float | None,
        b_max: float | None,
        clip: bool = False,
        relative: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.scaler = ScaleIntensityRangePercentiles(lower, upper, b_min, b_max, clip, relative, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
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
            match input data. the intensity values of input image corresponding
            to the selected values in the mask data will keep the original value,
            others will be set to `0`. if None, will extract the mask data from
            input data based on `mask_key`.
        mask_key: the key to extract mask data from input dictionary, only works
            when `mask_data` is None.
        select_fn: function to select valid values of the `mask_data`, default is
            to select `values > 0`.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = MaskIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_data: NdarrayOrTensor | None = None,
        mask_key: str | None = None,
        select_fn: Callable = is_positive,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = MaskIntensity(mask_data=mask_data, select_fn=select_fn)
        self.mask_key = mask_key if mask_data is None else None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], d[self.mask_key]) if self.mask_key is not None else self.converter(d[key])
        return d


class SavitzkyGolaySmoothd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SavitzkyGolaySmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        window_length: length of the filter window, must be a positive odd integer.
        order: order of the polynomial to fit to each window, must be less than ``window_length``.
        axis: optional axis along which to apply the filter kernel. Default 1 (first spatial dimension).
        mode: optional padding mode, passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. default: ``'zeros'``. See ``torch.nn.Conv1d()`` for more information.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = SavitzkyGolaySmooth.backend

    def __init__(
        self,
        keys: KeysCollection,
        window_length: int,
        order: int,
        axis: int = 1,
        mode: str = "zeros",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = SavitzkyGolaySmooth(window_length=window_length, order=order, axis=axis, mode=mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class MedianSmoothd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.MedianSmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        radius: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = MedianSmooth.backend

    def __init__(self, keys: KeysCollection, radius: Sequence[int] | int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = MedianSmooth(radius)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
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

    backend = GaussianSmooth.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma: Sequence[float] | float,
        approx: str = "erf",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GaussianSmooth(sigma, approx=approx)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
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

    backend = RandGaussianSmooth.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma_x: tuple[float, float] = (0.25, 1.5),
        sigma_y: tuple[float, float] = (0.25, 1.5),
        sigma_z: tuple[float, float] = (0.25, 1.5),
        approx: str = "erf",
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_smooth = RandGaussianSmooth(
            sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z, approx=approx, prob=1.0
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandGaussianSmoothd:
        super().set_random_state(seed, state)
        self.rand_smooth.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma
        self.rand_smooth.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.rand_smooth(d[key], randomize=False)
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

    backend = GaussianSharpen.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma1: Sequence[float] | float = 3.0,
        sigma2: Sequence[float] | float = 1.0,
        alpha: float = 30.0,
        approx: str = "erf",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GaussianSharpen(sigma1, sigma2, alpha, approx=approx)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
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

    backend = RandGaussianSharpen.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma1_x: tuple[float, float] = (0.5, 1.0),
        sigma1_y: tuple[float, float] = (0.5, 1.0),
        sigma1_z: tuple[float, float] = (0.5, 1.0),
        sigma2_x: tuple[float, float] | float = 0.5,
        sigma2_y: tuple[float, float] | float = 0.5,
        sigma2_z: tuple[float, float] | float = 0.5,
        alpha: tuple[float, float] = (10.0, 30.0),
        approx: str = "erf",
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_sharpen = RandGaussianSharpen(
            sigma1_x=sigma1_x,
            sigma1_y=sigma1_y,
            sigma1_z=sigma1_z,
            sigma2_x=sigma2_x,
            sigma2_y=sigma2_y,
            sigma2_z=sigma2_z,
            alpha=alpha,
            approx=approx,
            prob=1.0,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandGaussianSharpend:
        super().set_random_state(seed, state)
        self.rand_sharpen.set_random_state(seed, state)
        return self

    def __call__(self, data: dict[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma1, sigma2, etc.
        self.rand_sharpen.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.rand_sharpen(d[key], randomize=False)
        return d


class RandHistogramShiftd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandHistogramShift`.
    Apply random nonlinear transform the image's intensity histogram.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        num_control_points: number of control points governing the nonlinear intensity mapping.
            a smaller number of control points allows for larger intensity shifts. if two values provided, number of
            control points selecting from range (min_value, max_value).
        prob: probability of histogram shift.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandHistogramShift.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_control_points: tuple[int, int] | int = 10,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.shifter = RandHistogramShift(num_control_points=num_control_points, prob=1.0)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandHistogramShiftd:
        super().set_random_state(seed, state)
        self.shifter.set_random_state(seed, state)
        return self

    def __call__(self, data: dict[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random shift params
        self.shifter.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.shifter(d[key], randomize=False)
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
        alpha (float, Sequence[float]): Parametrizes the intensity of the Gibbs noise filter applied. Takes
            values in the interval [0,1] with alpha = 0 acting as the identity mapping.
            If a length-2 list is given as [a,b] then the value of alpha will be sampled
            uniformly from the interval [a,b].
            If a float is given, then the value of alpha will be sampled uniformly from the interval [0, alpha].
        allow_missing_keys: do not raise exception if key is missing.
    """

    backend = RandGibbsNoise.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        alpha: float | Sequence[float] = (0.0, 1.0),
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.rand_gibbs_noise = RandGibbsNoise(alpha=alpha, prob=1.0)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> RandGibbsNoised:
        super().set_random_state(seed, state)
        self.rand_gibbs_noise.set_random_state(seed, state)
        return self

    def __call__(self, data: dict[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random noise params
        self.rand_gibbs_noise.randomize(None)
        for key in self.key_iterator(d):
            d[key] = self.rand_gibbs_noise(d[key], randomize=False)
        return d


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
        allow_missing_keys: do not raise exception if key is missing.
    """

    backend = GibbsNoise.backend

    def __init__(self, keys: KeysCollection, alpha: float = 0.5, allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.transform = GibbsNoise(alpha)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class KSpaceSpikeNoised(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.KSpaceSpikeNoise`.

    Applies localized spikes in `k`-space at the given locations and intensities.
    Spike (Herringbone) artifact is a type of data acquisition artifact which
    may occur during MRI scans.

    For general information on spike artifacts, please refer to:

    `AAPM/RSNA physics tutorial for residents: fundamental physics of MR imaging
    <https://pubmed.ncbi.nlm.nih.gov/16009826>`_.

    `Body MRI artifacts in clinical practice: A physicist's and radiologist's
    perspective <https://doi.org/10.1002/jmri.24288>`_.

    Args:
        keys: "image", "label", or ["image", "label"] depending
             on which data you need to transform.
        loc: spatial location for the spikes. For
            images with 3D spatial dimensions, the user can provide (C, X, Y, Z)
            to fix which channel C is affected, or (X, Y, Z) to place the same
            spike in all channels. For 2D cases, the user can provide (C, X, Y)
            or (X, Y).
        k_intensity: value for the log-intensity of the
            `k`-space version of the image. If one location is passed to ``loc`` or the
            channel is not specified, then this argument should receive a float. If
            ``loc`` is given a sequence of locations, then this argument should
            receive a sequence of intensities. This value should be tested as it is
            data-dependent. The default values are the 2.5 the mean of the
            log-intensity for each channel.
        allow_missing_keys: do not raise exception if key is missing.

    Example:
        When working with 4D data,
        ``KSpaceSpikeNoised("image", loc = ((3,60,64,32), (64,60,32)), k_intensity = (13,14))``
        will place a spike at `[3, 60, 64, 32]` with `log-intensity = 13`, and
        one spike per channel located respectively at `[: , 64, 60, 32]`
        with `log-intensity = 14`.
    """

    backend = KSpaceSpikeNoise.backend

    def __init__(
        self,
        keys: KeysCollection,
        loc: tuple | Sequence[tuple],
        k_intensity: Sequence[float] | float | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = KSpaceSpikeNoise(loc, k_intensity)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: Expects image/label to have dimensions (C, H, W) or
                (C, H, W, D), where C is the channel.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class RandKSpaceSpikeNoised(RandomizableTransform, MapTransform):
    """
    Dictionary-based version of :py:class:`monai.transforms.RandKSpaceSpikeNoise`.

    Naturalistic data augmentation via spike artifacts. The transform applies
    localized spikes in `k`-space.

    For general information on spike artifacts, please refer to:

    `AAPM/RSNA physics tutorial for residents: fundamental physics of MR imaging
    <https://pubmed.ncbi.nlm.nih.gov/16009826>`_.

    `Body MRI artifacts in clinical practice: A physicist's and radiologist's
    perspective <https://doi.org/10.1002/jmri.24288>`_.

    Args:
        keys: "image", "label", or ["image", "label"] depending
             on which data you need to transform.
        prob: probability to add spike artifact to each item in the
            dictionary provided it is realized that the noise will be applied
            to the dictionary.
        intensity_range: pass a tuple (a, b) to sample the log-intensity from the interval (a, b)
            uniformly for all channels. Or pass sequence of intervals
            ((a0, b0), (a1, b1), ...) to sample for each respective channel.
            In the second case, the number of 2-tuples must match the number of channels.
            Default ranges is `(0.95x, 1.10x)` where `x` is the mean
            log-intensity for each channel.
        channel_wise: treat each channel independently. True by default.
        allow_missing_keys: do not raise exception if key is missing.

    Example:
        To apply `k`-space spikes randomly on the image only, with probability
        0.5, and log-intensity sampled from the interval [13, 15] for each
        channel independently, one uses
        ``RandKSpaceSpikeNoised("image", prob=0.5, intensity_ranges=(13, 15), channel_wise=True)``.
    """

    backend = RandKSpaceSpikeNoise.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        intensity_range: Sequence[Sequence[float] | float] | None = None,
        channel_wise: bool = True,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.rand_noise = RandKSpaceSpikeNoise(prob=1.0, intensity_range=intensity_range, channel_wise=channel_wise)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandKSpaceSpikeNoised:
        super().set_random_state(seed, state)
        self.rand_noise.set_random_state(seed, state)
        return self

    def __call__(self, data: dict[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        for key in self.key_iterator(d):
            d[key] = self.rand_noise(d[key], randomize=True)
        return d


class RandCoarseDropoutd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandCoarseDropout`.
    Expect all the data specified by `keys` have same spatial shape and will randomly dropout the same regions
    for every key, if want to dropout differently for every key, please use this transform separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        dropout_holes: if `True`, dropout the regions of holes and fill value, if `False`, keep the holes and
            dropout the outside and fill value. default to `True`.
        fill_value: target value to fill the dropout regions, if providing a number, will use it as constant
            value to fill all the regions. if providing a tuple for the `min` and `max`, will randomly select
            value for every pixel / voxel from the range `[min, max)`. if None, will compute the `min` and `max`
            value of input image then randomly select value to fill, default to None.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = RandCoarseDropout.backend

    def __init__(
        self,
        keys: KeysCollection,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.dropper = RandCoarseDropout(
            holes=holes,
            spatial_size=spatial_size,
            dropout_holes=dropout_holes,
            fill_value=fill_value,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
            prob=1.0,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCoarseDropoutd:
        super().set_random_state(seed, state)
        self.dropper.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.dropper.randomize(d[first_key].shape[1:])
        for key in self.key_iterator(d):
            d[key] = self.dropper(img=d[key], randomize=False)

        return d


class RandCoarseShuffled(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandCoarseShuffle`.
    Expect all the data specified by `keys` have same spatial shape and will randomly dropout the same regions
    for every key, if want to shuffle different regions for every key, please use this transform separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = RandCoarseShuffle.backend

    def __init__(
        self,
        keys: KeysCollection,
        holes: int,
        spatial_size: Sequence[int] | int,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.shuffle = RandCoarseShuffle(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=1.0
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCoarseShuffled:
        super().set_random_state(seed, state)
        self.shuffle.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.shuffle.randomize(d[first_key].shape[1:])
        for key in self.key_iterator(d):
            d[key] = self.shuffle(img=d[key], randomize=False)

        return d


class HistogramNormalized(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.HistogramNormalize`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        num_bins: number of the bins to use in histogram, default to `256`. for more details:
            https://numpy.org/doc/stable/reference/generated/numpy.histogram.html.
        min: the min value to normalize input image, default to `255`.
        max: the max value to normalize input image, default to `255`.
        mask: if provided, must be ndarray of bools or 0s and 1s, and same shape as `image`.
            only points at which `mask==True` are used for the equalization.
            can also provide the mask by `mask_key` at runtime.
        mask_key: if mask is None, will try to get the mask with `mask_key`.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: do not raise exception if key is missing.

    """

    backend = HistogramNormalize.backend

    def __init__(
        self,
        keys: KeysCollection,
        num_bins: int = 256,
        min: int = 0,
        max: int = 255,
        mask: NdarrayOrTensor | None = None,
        mask_key: str | None = None,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = HistogramNormalize(num_bins=num_bins, min=min, max=max, mask=mask, dtype=dtype)
        self.mask_key = mask_key if mask is None else None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], d[self.mask_key]) if self.mask_key is not None else self.transform(d[key])

        return d


class ForegroundMaskd(MapTransform):
    """
    Creates a binary mask that defines the foreground based on thresholds in RGB or HSV color space.
    This transform receives an RGB (or grayscale) image where by default it is assumed that the foreground has
    low values (dark) while the background is white.

    Args:
        keys: keys of the corresponding items to be transformed.
        threshold: an int or a float number that defines the threshold that values less than that are foreground.
            It also can be a callable that receives each dimension of the image and calculate the threshold,
            or a string that defines such callable from `skimage.filter.threshold_...`. For the list of available
            threshold functions, please refer to https://scikit-image.org/docs/stable/api/skimage.filters.html
            Moreover, a dictionary can be passed that defines such thresholds for each channel, like
            {"R": 100, "G": "otsu", "B": skimage.filter.threshold_mean}
        hsv_threshold: similar to threshold but HSV color space ("H", "S", and "V").
            Unlike RBG, in HSV, value greater than `hsv_threshold` are considered foreground.
        invert: invert the intensity range of the input image, so that the dtype maximum is now the dtype minimum,
            and vice-versa.
        new_key_prefix: this prefix be prepended to the key to create a new key for the output and keep the value of
            key intact. By default not prefix is set and the corresponding array to the key will be replaced.
        allow_missing_keys: do not raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        threshold: dict | Callable | str | float = "otsu",
        hsv_threshold: dict | Callable | str | float | int | None = None,
        invert: bool = False,
        new_key_prefix: str | None = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = ForegroundMask(threshold=threshold, hsv_threshold=hsv_threshold, invert=invert)
        self.new_key_prefix = new_key_prefix

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            new_key = key if self.new_key_prefix is None else self.new_key_prefix + key
            d[new_key] = self.transform(d[key])

        return d


class ComputeHoVerMapsd(MapTransform):
    """Compute horizontal and vertical maps from an instance mask
    It generates normalized horizontal and vertical distances to the center of mass of each region.

    Args:
        keys: keys of the corresponding items to be transformed.
        dtype: the type of output Tensor. Defaults to `"float32"`.
        new_key_prefix: this prefix be prepended to the key to create a new key for the output and keep the value of
            key intact. Defaults to '"_hover", so if the input key is "mask" the output will be "hover_mask".
        allow_missing_keys: do not raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = "float32",
        new_key_prefix: str = "hover_",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.transform = ComputeHoVerMaps(dtype=dtype)
        self.new_key_prefix = new_key_prefix

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            new_key = key if self.new_key_prefix is None else self.new_key_prefix + key
            d[new_key] = self.transform(d[key])

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
RandScaleIntensityFixedMeanD = RandScaleIntensityFixedMeanDict = RandScaleIntensityFixedMeand
NormalizeIntensityD = NormalizeIntensityDict = NormalizeIntensityd
ThresholdIntensityD = ThresholdIntensityDict = ThresholdIntensityd
ScaleIntensityRangeD = ScaleIntensityRangeDict = ScaleIntensityRanged
ClipIntensityPercentilesD = ClipIntensityPercentilesDict = ClipIntensityPercentilesd
AdjustContrastD = AdjustContrastDict = AdjustContrastd
RandAdjustContrastD = RandAdjustContrastDict = RandAdjustContrastd
ScaleIntensityRangePercentilesD = ScaleIntensityRangePercentilesDict = ScaleIntensityRangePercentilesd
MaskIntensityD = MaskIntensityDict = MaskIntensityd
SavitzkyGolaySmoothD = SavitzkyGolaySmoothDict = SavitzkyGolaySmoothd
MedianSmoothD = MedianSmoothDict = MedianSmoothd
GaussianSmoothD = GaussianSmoothDict = GaussianSmoothd
RandGaussianSmoothD = RandGaussianSmoothDict = RandGaussianSmoothd
GaussianSharpenD = GaussianSharpenDict = GaussianSharpend
RandGaussianSharpenD = RandGaussianSharpenDict = RandGaussianSharpend
RandHistogramShiftD = RandHistogramShiftDict = RandHistogramShiftd
RandGibbsNoiseD = RandGibbsNoiseDict = RandGibbsNoised
GibbsNoiseD = GibbsNoiseDict = GibbsNoised
KSpaceSpikeNoiseD = KSpaceSpikeNoiseDict = KSpaceSpikeNoised
RandKSpaceSpikeNoiseD = RandKSpaceSpikeNoiseDict = RandKSpaceSpikeNoised
RandCoarseDropoutD = RandCoarseDropoutDict = RandCoarseDropoutd
HistogramNormalizeD = HistogramNormalizeDict = HistogramNormalized
RandCoarseShuffleD = RandCoarseShuffleDict = RandCoarseShuffled
ForegroundMaskD = ForegroundMaskDict = ForegroundMaskd
ComputeHoVerMapsD = ComputeHoVerMapsDict = ComputeHoVerMapsd
