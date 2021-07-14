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
A collection of "vanilla" transforms for intensity adjustment
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import torch

from monai.config import DtypeLike
from monai.networks.layers import GaussianFilter, HilbertTransform, SavitzkyGolayFilter
from monai.transforms.transform import (
    RandomizableTransform,
    TorchOrNumpyTransform,
    TorchTransform,
    Transform,
    convert_data_type,
)
from monai.transforms.utils import rescale_array
from monai.utils import PT_BEFORE_1_7, InvalidPyTorchVersionError, ensure_tuple, ensure_tuple_rep, ensure_tuple_size
from monai.utils.enums import DataObjects
from monai.utils.misc import dtype_convert

__all__ = [
    "RandGaussianNoise",
    "RandRicianNoise",
    "ShiftIntensity",
    "RandShiftIntensity",
    "StdShiftIntensity",
    "RandStdShiftIntensity",
    "RandBiasField",
    "ScaleIntensity",
    "RandScaleIntensity",
    "NormalizeIntensity",
    "ThresholdIntensity",
    "ScaleIntensityRange",
    "AdjustContrast",
    "RandAdjustContrast",
    "ScaleIntensityRangePercentiles",
    "MaskIntensity",
    "DetectEnvelope",
    "SavitzkyGolaySmooth",
    "GaussianSmooth",
    "RandGaussianSmooth",
    "GaussianSharpen",
    "RandGaussianSharpen",
    "RandHistogramShift",
    "GibbsNoise",
    "RandGibbsNoise",
    "KSpaceSpikeNoise",
    "RandKSpaceSpikeNoise",
]


class RandGaussianNoise(RandomizableTransform):
    """
    Add Gaussian noise to image.

    Args:
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
    """

    def __init__(self, prob: float = 0.1, mean: Union[Sequence[float], float] = 0.0, std: float = 0.1) -> None:
        RandomizableTransform.__init__(self, prob)
        self.mean = mean
        self.std = std
        self._noise: np.ndarray

    def randomize(self, im_shape: Sequence[int]) -> None:
        super().randomize(None)
        self._noise = self.R.normal(self.mean, self.R.uniform(0, self.std), size=im_shape)

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        self.randomize(img.shape)
        if self._noise is None:
            raise AssertionError
        if not self._do_transform:
            return img
        dtype = dtype_convert(img.dtype, np.array)
        return img + self._noise.astype(dtype)


class RandRicianNoise(TorchOrNumpyTransform, RandomizableTransform):
    """
    Add Rician noise to image.
    Rician noise in MRI is the result of performing a magnitude operation on complex
    data with Gaussian noise of the same variance in both channels, as described in `Noise in Magnitude Magnetic Resonance Images
    <https://doi.org/10.1002/cmr.a.20124>`_. This transform is adapted from
    `DIPY<https://github.com/dipy/dipy>`_. See also: `The rician distribution of noisy mri data
    <https://doi.org/10.1002/mrm.1910340618>`_.

    Args:
        prob: Probability to add Rician noise.
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
    """

    def __init__(
        self,
        prob: float = 0.1,
        mean: Union[Sequence[float], float] = 0.0,
        std: Union[Sequence[float], float] = 1.0,
        channel_wise: bool = False,
        relative: bool = False,
        sample_std: bool = True,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.prob = prob
        self.mean = mean
        self.std = std
        self.channel_wise = channel_wise
        self.relative = relative
        self.sample_std = sample_std
        self._noise1: DataObjects.Images
        self._noise2: DataObjects.Images

    def _add_noise(self, img: DataObjects.Images, mean: float, std: float):
        im_shape = img.shape
        if isinstance(img, torch.Tensor):
            _std = float(torch.rand(1, generator=self.R_torch)) * std if self.sample_std else std
            self._noise1 = torch.normal(mean, _std, size=im_shape, generator=self.R_torch).to(img.dtype).to(img.device)
            self._noise2 = torch.normal(mean, _std, size=im_shape, generator=self.R_torch).to(img.dtype).to(img.device)
            return torch.sqrt((img + self._noise1) ** 2 + self._noise2 ** 2)
        else:
            _std = self.R.uniform(0, std) if self.sample_std else std
            self._noise1 = self.R.normal(mean, _std, size=im_shape).astype(img.dtype)
            self._noise2 = self.R.normal(mean, _std, size=im_shape).astype(img.dtype)
            return np.sqrt((img + self._noise1) ** 2 + self._noise2 ** 2)  # type: ignore

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        super().randomize(None)
        if not self._do_transform:
            return img
        if self.channel_wise:
            _mean = ensure_tuple_rep(self.mean, len(img))
            _std = ensure_tuple_rep(self.std, len(img))
            for i, d in enumerate(img):
                img[i] = self._add_noise(d, mean=_mean[i], std=_std[i] * d.std() if self.relative else _std[i])
        else:
            if not isinstance(self.mean, (int, float)):
                raise AssertionError("If channel_wise is False, mean must be a float or int number.")
            if not isinstance(self.std, (int, float)):
                raise AssertionError("If channel_wise is False, std must be a float or int number.")
            std = self.std * img.std() if self.relative else self.std
            if not isinstance(std, (int, float)):
                raise AssertionError
            img = self._add_noise(img, mean=self.mean, std=std)
        return img


class ShiftIntensity(TorchOrNumpyTransform):
    """
    Shift intensity uniformly for the entire image with specified `offset`.

    Args:
        offset: offset value to shift the intensity of image.
    """

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        out = img + self.offset
        if isinstance(out, torch.Tensor):
            return out.type(img.dtype)  # type: ignore
        return out.astype(img.dtype)  # type: ignore


class RandShiftIntensity(RandomizableTransform, TorchOrNumpyTransform):
    """
    Randomly shift intensity with randomly picked offset.
    """

    def __init__(self, offsets: Union[Tuple[float, float], float], prob: float = 0.1) -> None:
        """
        Args:
            offsets: offset range to randomly shift.
                if single number, offset value is picked from (-offsets, offsets).
            prob: probability of shift.
        """
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

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        self.randomize()
        if not self._do_transform:
            return img
        shifter = ShiftIntensity(self._offset)
        return shifter(img)


class StdShiftIntensity(TorchOrNumpyTransform):
    """
    Shift intensity for the image with a factor and the standard deviation of the image
    by: ``v = v + factor * std(v)``.
    This transform can focus on only non-zero values or the entire image,
    and can also calculate the std on each channel separately.

    Args:
        factor: factor shift by ``v = v + factor * std(v)``.
        nonzero: whether only count non-zero values.
        channel_wise: if True, calculate on each channel separately. Please ensure
            that the first dimension represents the channel of the image if True.
        dtype: output data type, defaults to float32.
    """

    def __init__(
        self, factor: float, nonzero: bool = False, channel_wise: bool = False, dtype: DtypeLike = np.float32
    ) -> None:
        self.factor = factor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    def _stdshift(self, img: DataObjects.Images) -> DataObjects.Images:
        ones: Callable
        std: Callable
        if isinstance(img, torch.Tensor):
            ones = torch.ones
            std = partial(torch.std, unbiased=False)
        else:
            ones = np.ones
            std = np.std

        slices = (img != 0) if self.nonzero else ones(img.shape, dtype=bool)
        if slices.any():
            out = deepcopy(img)
            offset = self.factor * std(out[slices])
            out[slices] = out[slices] + offset
            return out
        return img

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        img, *_ = convert_data_type(img, dtype=self.dtype)
        if self.channel_wise:
            for i, d in enumerate(img):
                img[i] = self._stdshift(d)  # type: ignore
        else:
            img = self._stdshift(img)
        return img


class RandStdShiftIntensity(TorchOrNumpyTransform, RandomizableTransform):
    """
    Shift intensity for the image with a factor and the standard deviation of the image
    by: ``v = v + factor * std(v)`` where the `factor` is randomly picked.
    """

    def __init__(
        self,
        factors: Union[Tuple[float, float], float],
        prob: float = 0.1,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        """
        Args:
            factors: if tuple, the randomly picked range is (min(factors), max(factors)).
                If single number, the range is (-factors, factors).
            prob: probability of std shift.
            nonzero: whether only count non-zero values.
            channel_wise: if True, calculate on each channel separately.
            dtype: output data type, defaults to float32.

        """
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

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        self.randomize()
        if not self._do_transform:
            return img
        shifter = StdShiftIntensity(
            factor=self.factor, nonzero=self.nonzero, channel_wise=self.channel_wise, dtype=self.dtype
        )
        return shifter(img)


class ScaleIntensity(TorchOrNumpyTransform):
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
            factor: factor scale by ``v = v * (1 + factor)``. In order to use
                this parameter, please set `minv` and `maxv` into None.
        """
        self.minv = minv
        self.maxv = maxv
        self.factor = factor

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.

        Raises:
            ValueError: When ``self.minv=None`` or ``self.maxv=None`` and ``self.factor=None``. Incompatible values.

        """
        if self.minv is not None and self.maxv is not None:
            return rescale_array(img, self.minv, self.maxv, img.dtype)
        if self.factor is not None:
            out = img * (1 + self.factor)
            if isinstance(out, torch.Tensor):
                return out.to(img.dtype)  # type: ignore
            else:
                return out.astype(img.dtype)  # type: ignore
        raise ValueError("Incompatible values: minv=None or maxv=None and factor=None.")


class RandScaleIntensity(TorchOrNumpyTransform, RandomizableTransform):
    """
    Randomly scale the intensity of input image by ``v = v * (1 + factor)`` where the `factor`
    is randomly picked.
    """

    def __init__(self, factors: Union[Tuple[float, float], float], prob: float = 0.1) -> None:
        """
        Args:
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of scale.

        """
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

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        self.randomize()
        if not self._do_transform:
            return img
        scaler = ScaleIntensity(minv=None, maxv=None, factor=self.factor)
        return scaler(img)


class RandBiasField(RandomizableTransform):
    """
    Random bias field augmentation for MR images.
    The bias field is considered as a linear combination of smoothly varying basis (polynomial)
    functions, as described in `Automated Model-Based Tissue Classification of MR Images of the Brain
    <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=811270>`_.
    This implementation adapted from `NiftyNet
    <https://github.com/NifTK/NiftyNet>`_.
    Referred to `Longitudinal segmentation of age-related white matter hyperintensities
    <https://www.sciencedirect.com/science/article/pii/S1361841517300257?via%3Dihub>`_.

    Args:
        degree: degree of freedom of the polynomials. The value should be no less than 1.
            Defaults to 3.
        coeff_range: range of the random coefficients. Defaults to (0.0, 0.1).
        dtype: output data type, defaults to float32.
        prob: probability to do random bias field.

    """

    def __init__(
        self,
        degree: int = 3,
        coeff_range: Tuple[float, float] = (0.0, 0.1),
        dtype: DtypeLike = np.float32,
        prob: float = 1.0,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        if degree < 1:
            raise ValueError("degree should be no less than 1.")
        self.degree = degree
        self.coeff_range = coeff_range
        self.dtype = dtype

    def _generate_random_field(
        self,
        spatial_shape: Tuple[int, ...],
        rank: int,
        degree: int,
        coeff: Tuple[int, ...],
    ):
        """
        products of polynomials as bias field estimations
        """
        coeff_mat = np.zeros((degree + 1,) * rank)
        coords = [np.linspace(-1.0, 1.0, dim, dtype=np.float32) for dim in spatial_shape]
        if rank == 2:
            coeff_mat[np.tril_indices(degree + 1)] = coeff
            field = np.polynomial.legendre.leggrid2d(coords[0], coords[1], coeff_mat)
        elif rank == 3:
            pts: List[List[int]] = [[0, 0, 0]]
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    for k in range(degree + 1 - i - j):
                        pts.append([i, j, k])
            if len(pts) > 1:
                pts = pts[1:]
            np_pts = np.stack(pts)
            coeff_mat[np_pts[:, 0], np_pts[:, 1], np_pts[:, 2]] = coeff
            field = np.polynomial.legendre.leggrid3d(coords[0], coords[1], coords[2], coeff_mat)
        else:
            raise NotImplementedError("only supports 2D or 3D fields")
        return field

    def randomize(self, data: np.ndarray) -> None:
        super().randomize(None)
        self.spatial_shape = data.shape[1:]
        self.rank = len(self.spatial_shape)
        n_coeff = int(np.prod([(self.degree + k) / k for k in range(1, self.rank + 1)]))
        self._coeff = self.R.uniform(*self.coeff_range, n_coeff).tolist()

    def __call__(self, img: np.ndarray):
        """
        Apply the transform to `img`.
        """
        self.randomize(data=img)
        if not self._do_transform:
            return img
        num_channels = img.shape[0]
        _bias_fields = np.stack(
            [
                self._generate_random_field(
                    spatial_shape=self.spatial_shape, rank=self.rank, degree=self.degree, coeff=self._coeff
                )
                for _ in range(num_channels)
            ],
            axis=0,
        )
        return (img * _bias_fields).astype(self.dtype)


class NormalizeIntensity(Transform):
    """
    Normalize input based on provided args, using calculated mean and std if not provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
        dtype: output data type, defaults to float32.
    """

    def __init__(
        self,
        subtrahend: Union[Sequence, np.ndarray, None] = None,
        divisor: Union[Sequence, np.ndarray, None] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype

    def _normalize(self, img: np.ndarray, sub=None, div=None) -> np.ndarray:
        slices = (img != 0) if self.nonzero else np.ones(img.shape, dtype=bool)
        if not np.any(slices):
            return img

        _sub = sub if sub is not None else np.mean(img[slices])
        if isinstance(_sub, np.ndarray):
            _sub = _sub[slices]

        _div = div if div is not None else np.std(img[slices])
        if np.isscalar(_div):
            if _div == 0.0:
                _div = 1.0
        elif isinstance(_div, np.ndarray):
            _div = _div[slices]
            _div[_div == 0.0] = 1.0
        img[slices] = (img[slices] - _sub) / _div
        return img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        if self.channel_wise:
            if self.subtrahend is not None and len(self.subtrahend) != len(img):
                raise ValueError(f"img has {len(img)} channels, but subtrahend has {len(self.subtrahend)} components.")
            if self.divisor is not None and len(self.divisor) != len(img):
                raise ValueError(f"img has {len(img)} channels, but divisor has {len(self.divisor)} components.")

            for i, d in enumerate(img):
                img[i] = self._normalize(
                    d,
                    sub=self.subtrahend[i] if self.subtrahend is not None else None,
                    div=self.divisor[i] if self.divisor is not None else None,
                )
        else:
            img = self._normalize(img, self.subtrahend, self.divisor)

        return img.astype(self.dtype)


class ThresholdIntensity(Transform):
    """
    Filter the intensity values of whole image to below threshold or above threshold.
    And fill the remaining parts of the image to the `cval` value.

    Args:
        threshold: the threshold to filter intensity values.
        above: filter values above the threshold or below the threshold, default is True.
        cval: value to fill the remaining parts of the image, default is 0.
    """

    def __init__(self, threshold: float, above: bool = True, cval: float = 0.0) -> None:
        if not isinstance(threshold, (int, float)):
            raise AssertionError("threshold must be a float or int number.")
        self.threshold = threshold
        self.above = above
        self.cval = cval

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        return np.asarray(
            np.where(img > self.threshold if self.above else img < self.threshold, img, self.cval), dtype=img.dtype
        )


class ScaleIntensityRange(TorchOrNumpyTransform):
    """
    Apply specific intensity scaling to the whole numpy array.
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

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        if self.a_max - self.a_min == 0.0:
            warn("Divide by zero (a_min == a_max)", Warning)
            return img - self.a_min + self.b_min

        img = (img - self.a_min) / (self.a_max - self.a_min)
        img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            mod = torch if isinstance(img, torch.Tensor) else np
            img = mod.clip(img, self.b_min, self.b_max)  # type: ignore
        return img


class AdjustContrast(TorchOrNumpyTransform):
    """
    Changes image intensity by gamma. Each pixel/voxel intensity is updated as::

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        gamma: gamma value to adjust the contrast as function.
    """

    def __init__(self, gamma: float) -> None:
        if not isinstance(gamma, (int, float)):
            raise AssertionError("gamma must be a float or int number.")
        self.gamma = gamma

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        epsilon = 1e-7
        img_min = img.min()
        img_range = img.max() - img_min
        return ((img - img_min) / float(img_range + epsilon)) ** self.gamma * img_range + img_min


class RandAdjustContrast(TorchOrNumpyTransform, RandomizableTransform):
    """
    Randomly changes image intensity by gamma. Each pixel/voxel intensity is updated as::

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
    """

    def __init__(self, prob: float = 0.1, gamma: Union[Sequence[float], float] = (0.5, 4.5)) -> None:
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

        self.gamma_value: float

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.gamma_value = self.R.uniform(low=self.gamma[0], high=self.gamma[1])

    def __call__(self, img: np.ndarray):
        """
        Apply the transform to `img`.
        """
        self.randomize()
        if self.gamma_value is None:
            raise AssertionError
        if not self._do_transform:
            return img
        adjuster = AdjustContrast(self.gamma_value)
        return adjuster(img)


class ScaleIntensityRangePercentiles(TorchOrNumpyTransform):
    """
    Apply range scaling to a numpy array based on the intensity distribution of the input.

    By default this transform will scale from [lower_intensity_percentile, upper_intensity_percentile] to [b_min, b_max], where
    {lower,upper}_intensity_percentile are the intensity values at the corresponding percentiles of ``img``.

    The ``relative`` parameter can also be set to scale from [lower_intensity_percentile, upper_intensity_percentile] to the
    lower and upper percentiles of the output range [b_min, b_max]

    For example:

    .. code-block:: python
        :emphasize-lines: 11, 22

        image = np.array(
            [[[1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]]])

        # Scale from lower and upper image intensity percentiles
        # to output range [b_min, b_max]
        scaler = ScaleIntensityRangePercentiles(10, 90, 0, 200, False, False)
        print(scaler(image))
        [[[0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.],
          [0., 50., 100., 150., 200.]]]

        # Scale from lower and upper image intensity percentiles
        # to lower and upper percentiles of the output range [b_min, b_max]
        rel_scaler = ScaleIntensityRangePercentiles(10, 90, 0, 200, False, True)
        print(rel_scaler(image))
        [[[20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.],
          [20., 60., 100., 140., 180.]]]


    Args:
        lower: lower intensity percentile.
        upper: upper intensity percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        relative: whether to scale to the corresponding percentiles of [b_min, b_max].
    """

    def __init__(
        self, lower: float, upper: float, b_min: float, b_max: float, clip: bool = False, relative: bool = False
    ) -> None:
        if lower < 0.0 or lower > 100.0:
            raise AssertionError("Percentiles must be in the range [0, 100]")
        if upper < 0.0 or upper > 100.0:
            raise AssertionError("Percentiles must be in the range [0, 100]")
        self.lower = lower
        self.upper = upper
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.relative = relative

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Apply the transform to `img`.
        """
        a_min = np.percentile(img, self.lower)
        a_max = np.percentile(img, self.upper)
        b_min = self.b_min
        b_max = self.b_max

        if self.relative:
            b_min = ((self.b_max - self.b_min) * (self.lower / 100.0)) + self.b_min
            b_max = ((self.b_max - self.b_min) * (self.upper / 100.0)) + self.b_min

        scalar = ScaleIntensityRange(a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=False)
        img = scalar(img)

        if self.clip:
            mod = torch if isinstance(img, torch.Tensor) else np
            img = mod.clip(img, self.b_min, self.b_max)  # type: ignore

        return img


class MaskIntensity(TorchOrNumpyTransform):
    """
    Mask the intensity values of input image with the specified mask data.
    Mask data must have the same spatial size as the input image, and all
    the intensity values of input image corresponding to `0` in the mask
    data will be set to `0`, others will keep the original value.

    Args:
        mask_data: if `mask_data` is single channel, apply to every channel
            of input image. if multiple channels, the number of channels must
            match the input data. `mask_data` will be converted to `bool` values
            by `mask_data > 0` before applying transform to input image.

    """

    def __init__(self, mask_data: Optional[DataObjects.Images]) -> None:
        self.mask_data = mask_data

    def __call__(self, img: DataObjects.Images, mask_data: Optional[DataObjects.Images] = None) -> DataObjects.Images:
        """
        Args:
            mask_data: if mask data is single channel, apply to every channel
                of input image. if multiple channels, the channel number must
                match input data. mask_data will be converted to `bool` values
                by `mask_data > 0` before applying transform to input image.

        Raises:
            - ValueError: When both ``mask_data`` and ``self.mask_data`` are None.
            - ValueError: When ``mask_data`` and ``img`` channels differ and ``mask_data`` is not single channel.

        """

        if mask_data is not None:
            mask_data_ = mask_data > 0
        elif self.mask_data is not None:
            mask_data_ = self.mask_data > 0
        else:
            raise ValueError("Unknown mask_data.")
        if mask_data_.shape[0] != 1 and mask_data_.shape[0] != img.shape[0]:
            raise ValueError(
                "When mask_data is not single channel, mask_data channels must match img, "
                f"got img={img.shape[0]} mask_data={mask_data_.shape[0]}."
            )
        mask_data_, *_ = convert_data_type(
            mask_data_, type(img), device=img.device if isinstance(img, torch.Tensor) else None
        )
        return img * mask_data_


class SavitzkyGolaySmooth(TorchTransform):
    """
    Smooth the input data along the given axis using a Savitzky-Golay filter.

    Args:
        window_length: Length of the filter window, must be a positive odd integer.
        order: Order of the polynomial to fit to each window, must be less than ``window_length``.
        axis: Optional axis along which to apply the filter kernel. Default 1 (first spatial dimension).
        mode: Optional padding mode, passed to convolution class. ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``. See ``torch.nn.Conv1d()`` for more information.
    """

    def __init__(self, window_length: int, order: int, axis: int = 1, mode: str = "zeros"):

        if axis < 0:
            raise ValueError("axis must be zero or positive.")

        self.window_length = window_length
        self.order = order
        self.axis = axis
        self.mode = mode

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: array containing input data. Must be real and in shape [channels, spatial1, spatial2, ...].

        Returns:
            array containing smoothed result.

        """
        img_t: torch.Tensor
        img_t, orig_type, orig_device = self.pre_conv_data(img)  # type: ignore

        # add one to transform axis because a batch axis will be added at dimension 0
        savgol_filter = SavitzkyGolayFilter(self.window_length, self.order, self.axis + 1, self.mode)
        # convert to Tensor and add Batch axis expected by HilbertTransform
        out = savgol_filter(img_t.unsqueeze(0)).squeeze(0)
        return self.post_convert_data(out, orig_type, orig_device)


class DetectEnvelope(TorchTransform):
    """
    Find the envelope of the input data along the requested axis using a Hilbert transform.
    Requires PyTorch 1.7.0+ and the PyTorch FFT module (which is not included in NVIDIA PyTorch Release 20.10).

    Args:
        axis: Axis along which to detect the envelope. Default 1, i.e. the first spatial dimension.
        N: FFT size. Default img.shape[axis]. Input will be zero-padded or truncated to this size along dimension
        ``axis``.

    """

    def __init__(self, axis: int = 1, n: Union[int, None] = None) -> None:

        if PT_BEFORE_1_7:
            raise InvalidPyTorchVersionError("1.7.0", self.__class__.__name__)

        if axis < 0:
            raise ValueError("axis must be zero or positive.")

        self.axis = axis
        self.n = n

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """

        Args:
            img: array containing input data. Must be real and in shape [channels, spatial1, spatial2, ...].

        Returns:
            array containing envelope of data in img along the specified axis.

        """
        img_t: torch.Tensor
        img_t, orig_type, orig_device = self.pre_conv_data(img)  # type: ignore

        # add one to transform axis because a batch axis will be added at dimension 0
        hilbert_transform = HilbertTransform(self.axis + 1, self.n)
        # convert to Tensor and add Batch axis expected by HilbertTransform
        out = torch.abs(hilbert_transform(img_t.unsqueeze(0))).squeeze(0)
        return self.post_convert_data(out, orig_type, orig_device)


class GaussianSmooth(TorchTransform):
    """
    Apply Gaussian smooth to the input data based on specified `sigma` parameter.
    A default value `sigma=1.0` is provided for reference.

    Args:
        sigma: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.

    """

    def __init__(self, sigma: Union[Sequence[float], float] = 1.0, approx: str = "erf") -> None:
        self.sigma = sigma
        self.approx = approx

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        img_t: torch.Tensor
        img_t, orig_type, orig_device = self.pre_conv_data(img)  # type: ignore
        img_t = img_t.to(torch.float)

        gaussian_filter = GaussianFilter(img.ndim - 1, self.sigma, approx=self.approx).to(img_t.device)
        out = gaussian_filter(img_t.unsqueeze(0)).squeeze(0)
        return self.post_convert_data(out, orig_type, orig_device)


class RandGaussianSmooth(TorchTransform, RandomizableTransform):
    """
    Apply Gaussian smooth to the input data based on randomly selected `sigma` parameters.

    Args:
        sigma_x: randomly select sigma value for the first spatial dimension.
        sigma_y: randomly select sigma value for the second spatial dimension if have.
        sigma_z: randomly select sigma value for the third spatial dimension if have.
        prob: probability of Gaussian smooth.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.

    """

    def __init__(
        self,
        sigma_x: Tuple[float, float] = (0.25, 1.5),
        sigma_y: Tuple[float, float] = (0.25, 1.5),
        sigma_z: Tuple[float, float] = (0.25, 1.5),
        prob: float = 0.1,
        approx: str = "erf",
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.approx = approx

        self.x = self.sigma_x[0]
        self.y = self.sigma_y[0]
        self.z = self.sigma_z[0]

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.x = self.R.uniform(low=self.sigma_x[0], high=self.sigma_x[1])
        self.y = self.R.uniform(low=self.sigma_y[0], high=self.sigma_y[1])
        self.z = self.R.uniform(low=self.sigma_z[0], high=self.sigma_z[1])

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        self.randomize()
        if not self._do_transform:
            return img
        sigma = ensure_tuple_size(tup=(self.x, self.y, self.z), dim=img.ndim - 1)
        return GaussianSmooth(sigma=sigma, approx=self.approx)(img)


class GaussianSharpen(TorchTransform):
    """
    Sharpen images using the Gaussian Blur filter.
    Referring to: http://scipy-lectures.org/advanced/image_processing/auto_examples/plot_sharpen.html.
    The algorithm is shown as below

    .. code-block:: python

        blurred_f = gaussian_filter(img, sigma1)
        filter_blurred_f = gaussian_filter(blurred_f, sigma2)
        img = blurred_f + alpha * (blurred_f - filter_blurred_f)

    A set of default values `sigma1=3.0`, `sigma2=1.0` and `alpha=30.0` is provide for reference.

    Args:
        sigma1: sigma parameter for the first gaussian kernel. if a list of values, must match the count
            of spatial dimensions of input data, and apply every value in the list to 1 spatial dimension.
            if only 1 value provided, use it for all spatial dimensions.
        sigma2: sigma parameter for the second gaussian kernel. if a list of values, must match the count
            of spatial dimensions of input data, and apply every value in the list to 1 spatial dimension.
            if only 1 value provided, use it for all spatial dimensions.
        alpha: weight parameter to compute the final result.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.

    """

    def __init__(
        self,
        sigma1: Union[Sequence[float], float] = 3.0,
        sigma2: Union[Sequence[float], float] = 1.0,
        alpha: float = 30.0,
        approx: str = "erf",
    ) -> None:
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alpha = alpha
        self.approx = approx

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        img_t: torch.Tensor
        img_t, orig_type, orig_device = self.pre_conv_data(img)  # type: ignore
        img_t = img_t.to(torch.float)

        gf1, gf2 = [
            GaussianFilter(img_t.ndim - 1, sigma, approx=self.approx).to(img_t.device)
            for sigma in (self.sigma1, self.sigma2)
        ]
        blurred_f = gf1(img_t.unsqueeze(0))
        filter_blurred_f = gf2(blurred_f)
        out = (blurred_f + self.alpha * (blurred_f - filter_blurred_f)).squeeze(0)
        return self.post_convert_data(out, orig_type, orig_device)


class RandGaussianSharpen(TorchTransform, RandomizableTransform):
    """
    Sharpen images using the Gaussian Blur filter based on randomly selected `sigma1`, `sigma2` and `alpha`.
    The algorithm is :py:class:`monai.transforms.GaussianSharpen`.

    Args:
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

    """

    def __init__(
        self,
        sigma1_x: Tuple[float, float] = (0.5, 1.0),
        sigma1_y: Tuple[float, float] = (0.5, 1.0),
        sigma1_z: Tuple[float, float] = (0.5, 1.0),
        sigma2_x: Union[Tuple[float, float], float] = 0.5,
        sigma2_y: Union[Tuple[float, float], float] = 0.5,
        sigma2_z: Union[Tuple[float, float], float] = 0.5,
        alpha: Tuple[float, float] = (10.0, 30.0),
        approx: str = "erf",
        prob: float = 0.1,
    ) -> None:
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

    def __call__(self, img: np.ndarray):
        self.randomize()
        if not self._do_transform:
            return img
        sigma1 = ensure_tuple_size(tup=(self.x1, self.y1, self.z1), dim=img.ndim - 1)
        sigma2 = ensure_tuple_size(tup=(self.x2, self.y2, self.z2), dim=img.ndim - 1)
        return GaussianSharpen(sigma1=sigma1, sigma2=sigma2, alpha=self.a, approx=self.approx)(img)


class RandHistogramShift(RandomizableTransform):
    """
    Apply random nonlinear transform to the image's intensity histogram.

    Args:
        num_control_points: number of control points governing the nonlinear intensity mapping.
            a smaller number of control points allows for larger intensity shifts. if two values provided, number of
            control points selecting from range (min_value, max_value).
        prob: probability of histogram shift.
    """

    def __init__(self, num_control_points: Union[Tuple[int, int], int] = 10, prob: float = 0.1) -> None:
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

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self.randomize()
        if not self._do_transform:
            return img
        img_min, img_max = img.min(), img.max()
        reference_control_points_scaled = self.reference_control_points * (img_max - img_min) + img_min
        floating_control_points_scaled = self.floating_control_points * (img_max - img_min) + img_min
        return np.asarray(
            np.interp(img, reference_control_points_scaled, floating_control_points_scaled), dtype=img.dtype
        )


class RandGibbsNoise(TorchOrNumpyTransform, RandomizableTransform):
    """
    Naturalistic image augmentation via Gibbs artifacts. The transform
    randomly applies Gibbs noise to 2D/3D MRI images. Gibbs artifacts
    are one of the common type of type artifacts appearing in MRI scans.

    The transform is applied to all the channels in the data.

    For general information on Gibbs artifacts, please refer to:
    https://pubs.rsna.org/doi/full/10.1148/rg.313105115
    https://pubs.rsna.org/doi/full/10.1148/radiographics.22.4.g02jl14949


    Args:
        prob (float): probability of applying the transform.
        alpha (float, Sequence(float)): Parametrizes the intensity of the Gibbs noise filter applied. Takes
            values in the interval [0,1] with alpha = 0 acting as the identity mapping.
            If a length-2 list is given as [a,b] then the value of alpha will be
            sampled uniformly from the interval [a,b]. 0 <= a <= b <= 1.
    """

    def __init__(self, prob: float = 0.1, alpha: Sequence[float] = (0.0, 1.0)) -> None:

        if len(alpha) != 2:
            raise AssertionError("alpha length must be 2.")
        if alpha[1] > 1 or alpha[0] < 0:
            raise AssertionError("alpha must take values in the interval [0,1]")
        if alpha[0] > alpha[1]:
            raise AssertionError("When alpha = [a,b] we need a < b.")

        self.alpha = alpha
        self.sampled_alpha = -1.0  # stores last alpha sampled by randomize()

        RandomizableTransform.__init__(self, prob=prob)

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:

        # randomize application and possibly alpha
        self._randomize(None)

        if self._do_transform:
            # apply transform
            transform = GibbsNoise(self.sampled_alpha)
            img = transform(img)
        return img

    def _randomize(self, _: Any) -> None:
        """
        (1) Set random variable to apply the transform.
        (2) Get alpha from uniform distribution.
        """
        super().randomize(None)
        self.sampled_alpha = self.R.uniform(self.alpha[0], self.alpha[1])


class GibbsNoise(TorchOrNumpyTransform):
    """
    The transform applies Gibbs noise to 2D/3D MRI images. Gibbs artifacts
    are one of the common type of type artifacts appearing in MRI scans.

    The transform is applied to all the channels in the data.

    For general information on Gibbs artifacts, please refer to:
    https://pubs.rsna.org/doi/full/10.1148/rg.313105115
    https://pubs.rsna.org/doi/full/10.1148/radiographics.22.4.g02jl14949


    Args:
        alpha (float): Parametrizes the intensity of the Gibbs noise filter applied. Takes
            values in the interval [0,1] with alpha = 0 acting as the identity mapping.
    """

    def __init__(self, alpha: float = 0.5) -> None:

        if alpha > 1 or alpha < 0:
            raise AssertionError("alpha must take values in the interval [0,1].")
        self.alpha = alpha

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        n_dims = len(img.shape[1:])
        data_type = type(img)
        # FT
        k = self._shift_fourier(img, n_dims, data_type)
        # build and apply mask
        k = self._apply_mask(k, data_type)
        # map back
        return self._inv_shift_fourier(k, n_dims, data_type)

    def _shift_fourier(self, x: DataObjects.Images, n_dims: int, data_type: type) -> DataObjects.Images:
        """
        Applies fourier transform and shifts its output.
        Only the spatial dimensions get transformed.

        Args:
            x: tensor to fourier transform.
        """
        # argument is dim if torch, else axes
        mod, arg = (torch, "dim") if data_type is torch.Tensor else (np, "axes")
        arg_dict = {arg: tuple(range(-n_dims, 0))}
        out: DataObjects.Images = mod.fft.fftshift(mod.fft.fftn(x, **arg_dict), **arg_dict)  # type: ignore
        return out

    def _inv_shift_fourier(self, k: DataObjects.Images, n_dims: int, data_type: type) -> DataObjects.Images:
        """
        Applies inverse shift and fourier transform. Only the spatial
        dimensions are transformed.
        """
        dims = tuple(range(-n_dims, 0))
        out: DataObjects.Images
        if data_type is torch.Tensor:
            out = torch.fft.ifftn(torch.fft.ifftshift(k, dim=dims), dim=dims, norm="backward")
        else:
            out = np.fft.ifftn(np.fft.ifftshift(k, axes=dims), axes=dims)
        return out.real

    def _apply_mask(self, k: DataObjects.Images, data_type: type) -> DataObjects.Images:
        """Builds and applies a mask on the spatial dimensions.

        Args:
            k: k-space version of the image.
        Returns:
            masked version of the k-space image.
        """
        shape = k.shape[1:]

        # compute masking radius and center
        r = (1 - self.alpha) * np.max(shape) * np.sqrt(2) / 2.0
        center = (np.array(shape) - 1) / 2

        # gives list w/ len==self.dim. Each dim gives coordinate in that dimension
        coords = np.ogrid[tuple(slice(0, i) for i in shape)]

        # need to subtract center coord and then square for Euc distance
        coords_from_center_sq = [(coord - c) ** 2 for coord, c in zip(coords, center)]
        dist_from_center = np.sqrt(sum(coords_from_center_sq))
        mask = dist_from_center <= r

        # add channel dimension into mask
        mask = np.repeat(mask[None], k.shape[0], axis=0)

        if data_type is torch.Tensor:
            mask = torch.Tensor(mask).to(k.device)  # type: ignore

        # apply binary mask
        out: DataObjects.Images = k * mask  # type: ignore
        return out


class KSpaceSpikeNoise(Transform):
    """
    Apply localized spikes in `k`-space at the given locations and intensities.
    Spike (Herringbone) artifact is a type of data acquisition artifact which
    may occur during MRI scans.

    For general information on spike artifacts, please refer to:

    `AAPM/RSNA physics tutorial for residents: fundamental physics of MR imaging
    <https://pubmed.ncbi.nlm.nih.gov/16009826>`_.

    `Body MRI artifacts in clinical practice: A physicist's and radiologist's
    perspective <https://doi.org/10.1002/jmri.24288>`_.

    Args:
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
        as_tensor_output: if ``True`` return torch.Tensor, else return np.array.
            Default: ``True``.

    Example:
        When working with 4D data, ``KSpaceSpikeNoise(loc = ((3,60,64,32), (64,60,32)), k_intensity = (13,14))``
        will place a spike at `[3, 60, 64, 32]` with `log-intensity = 13`, and
        one spike per channel located respectively at `[: , 64, 60, 32]`
        with `log-intensity = 14`.
    """

    def __init__(
        self,
        loc: Union[Tuple, Sequence[Tuple]],
        k_intensity: Optional[Union[Sequence[float], float]] = None,
        as_tensor_output: bool = True,
    ):

        self.loc = ensure_tuple(loc)
        self.as_tensor_output = as_tensor_output
        self.k_intensity = k_intensity

        # assert one-to-one relationship between factors and locations
        if isinstance(k_intensity, Sequence):
            if not isinstance(loc[0], Sequence):
                raise AssertionError(
                    "If a sequence is passed to k_intensity, then a sequence of locations must be passed to loc"
                )
            elif len(k_intensity) != len(loc):
                raise AssertionError("There must be one intensity_factor value for each tuple of indices in loc.")
        if isinstance(self.loc[0], Sequence) and k_intensity is not None:
            if not isinstance(self.k_intensity, Sequence):
                raise AssertionError("There must be one intensity_factor value for each tuple of indices in loc.")

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]:
        """
        Args:
            img (np.array or torch.tensor): image with dimensions (C, H, W) or (C, H, W, D)
        """
        # checking that tuples in loc are consistent with img size
        self._check_indices(img)

        if len(img.shape) < 3:
            raise AssertionError("Image needs a channel direction.")
        if isinstance(self.loc[0], int) and len(img.shape) == 4 and len(self.loc) == 2:
            raise AssertionError("Input images of dimension 4 need location tuple to be length 3 or 4")
        if isinstance(self.loc[0], Sequence) and len(img.shape) == 4 and min(map(lambda x: len(x), self.loc)) == 2:
            raise AssertionError("Input images of dimension 4 need location tuple to be length 3 or 4")

        n_dims = len(img.shape[1:])

        # convert to ndarray to work with np.fft
        if isinstance(img, torch.Tensor):
            device = img.device
            img = img.cpu().detach().numpy()
        else:
            device = torch.device("cpu")

        # FT
        k = self._shift_fourier(img, n_dims)
        log_abs = np.log(np.absolute(k) + 1e-10)
        phase = np.angle(k)

        k_intensity = self.k_intensity
        # default log intensity
        if k_intensity is None:
            k_intensity = tuple(np.mean(log_abs, axis=tuple(range(-n_dims, 0))) * 2.5)

        # highlight
        if isinstance(self.loc[0], Sequence):
            for idx, val in zip(self.loc, ensure_tuple(k_intensity)):
                self._set_spike(log_abs, idx, val)
        else:
            self._set_spike(log_abs, self.loc, k_intensity)
        # map back
        k = np.exp(log_abs) * np.exp(1j * phase)
        img = self._inv_shift_fourier(k, n_dims)
        return torch.Tensor(img, device=device) if self.as_tensor_output else img

    def _check_indices(self, img) -> None:
        """Helper method to check consistency of self.loc and input image.

        Raises assertion error if any index in loc is out of bounds."""

        loc = list(self.loc)
        if not isinstance(loc[0], Sequence):
            loc = [loc]
        for i in range(len(loc)):
            if len(loc[i]) < len(img.shape):
                loc[i] = [0] + list(loc[i])

        for i in range(len(img.shape)):
            if img.shape[i] <= max([x[i] for x in loc]):
                raise AssertionError(
                    f"The index value at position {i} of one of the tuples in loc = {self.loc} is out of bounds for current image."
                )

    def _set_spike(self, k: np.ndarray, idx: Tuple, val: Union[Sequence[float], float]):
        """
        Helper function to introduce a given intensity at given location.

        Args:
            k (np.array): intensity array to alter.
            idx (tuple): index of location where to apply change.
            val (float): value of intensity to write in.
        """
        if len(k.shape) == len(idx):
            if isinstance(val, Sequence):
                k[idx] = val[idx[0]]
            else:
                k[idx] = val
        elif len(k.shape) == 4 and len(idx) == 3:
            k[:, idx[0], idx[1], idx[2]] = val
        elif len(k.shape) == 3 and len(idx) == 2:
            k[:, idx[0], idx[1]] = val

    def _shift_fourier(self, x: Union[np.ndarray, torch.Tensor], n_dims: int) -> np.ndarray:
        """
        Applies fourier transform and shifts its output.
        Only the spatial dimensions get transformed.

        Args:
            x (np.ndarray): tensor to fourier transform.
        """
        out: np.ndarray = np.fft.fftshift(np.fft.fftn(x, axes=tuple(range(-n_dims, 0))), axes=tuple(range(-n_dims, 0)))
        return out

    def _inv_shift_fourier(self, k: Union[np.ndarray, torch.Tensor], n_dims: int) -> np.ndarray:
        """
        Applies inverse shift and fourier transform. Only the spatial
        dimensions are transformed.
        """
        out: np.ndarray = np.fft.ifftn(
            np.fft.ifftshift(k, axes=tuple(range(-n_dims, 0))), axes=tuple(range(-n_dims, 0))
        ).real
        return out


class RandKSpaceSpikeNoise(RandomizableTransform):
    """
    Naturalistic data augmentation via spike artifacts. The transform applies
    localized spikes in `k`-space, and it is the random version of
    :py:class:`monai.transforms.KSpaceSpikeNoise`.

    Spike (Herringbone) artifact is a type of data acquisition artifact which
    may occur during MRI scans. For general information on spike artifacts,
    please refer to:

    `AAPM/RSNA physics tutorial for residents: fundamental physics of MR imaging
    <https://pubmed.ncbi.nlm.nih.gov/16009826>`_.

    `Body MRI artifacts in clinical practice: A physicist's and radiologist's
    perspective <https://doi.org/10.1002/jmri.24288>`_.

    Args:
        prob: probability of applying the transform, either on all
            channels at once, or channel-wise if ``channel_wise = True``.
        intensity_range: pass a tuple
            (a, b) to sample the log-intensity from the interval (a, b)
            uniformly for all channels. Or pass sequence of intervals
            ((a0, b0), (a1, b1), ...) to sample for each respective channel.
            In the second case, the number of 2-tuples must match the number of
            channels.
            Default ranges is `(0.95x, 1.10x)` where `x` is the mean
            log-intensity for each channel.
        channel_wise: treat each channel independently. True by
            default.
        as_tensor_output: if True return torch.Tensor, else
            return np.array. default: True.

    Example:
        To apply `k`-space spikes randomly with probability 0.5, and
        log-intensity sampled from the interval [11, 12] for each channel
        independently, one uses
        ``RandKSpaceSpikeNoise(prob=0.5, intensity_range=(11, 12), channel_wise=True)``
    """

    def __init__(
        self,
        prob: float = 0.1,
        intensity_range: Optional[Sequence[Union[Sequence[float], float]]] = None,
        channel_wise=True,
        as_tensor_output: bool = True,
    ):

        self.intensity_range = intensity_range
        self.channel_wise = channel_wise
        self.as_tensor_output = as_tensor_output
        self.sampled_k_intensity: List[float] = []
        self.sampled_locs: List[Tuple] = []

        if intensity_range is not None:
            if isinstance(intensity_range[0], Sequence) and not channel_wise:
                raise AssertionError(
                    "When channel_wise = False, intensity_range should be a 2-tuple (low, high) or None."
                )

        super().__init__(prob)

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply transform to `img`. Assumes data is in channel-first form.

        Args:
            img (np.array or torch.tensor): image with dimensions (C, H, W) or (C, H, W, D)
        """
        if self.intensity_range is not None:
            if isinstance(self.intensity_range[0], Sequence) and len(self.intensity_range) != img.shape[0]:
                raise AssertionError(
                    "If intensity_range is a sequence of sequences, then there must be one (low, high) tuple for each channel."
                )

        self.sampled_k_intensity = []
        self.sampled_locs = []

        # convert to ndarray to work with np.fft
        x, device = self._to_numpy(img)
        intensity_range = self._make_sequence(x)
        self._randomize(x, intensity_range)

        # build/apply transform only if there are spike locations
        if self.sampled_locs:
            transform = KSpaceSpikeNoise(self.sampled_locs, self.sampled_k_intensity, self.as_tensor_output)
            return transform(x)

        return torch.Tensor(x, device=device) if self.as_tensor_output else x

    def _randomize(self, img: np.ndarray, intensity_range: Sequence[Sequence[float]]) -> None:
        """
        Helper method to sample both the location and intensity of the spikes.
        When not working channel wise (channel_wise=False) it use the random
        variable ``self._do_transform`` to decide whether to sample a location
        and intensity.

        When working channel wise, the method randomly samples a location and
        intensity for each channel depending on ``self._do_transform``.
        """
        # randomizing per channel
        if self.channel_wise:
            for i, chan in enumerate(img):
                super().randomize(None)
                if self._do_transform:
                    self.sampled_locs.append((i,) + tuple(self.R.randint(0, k) for k in chan.shape))
                    self.sampled_k_intensity.append(self.R.uniform(intensity_range[i][0], intensity_range[i][1]))
        # working with all channels together
        else:
            super().randomize(None)
            if self._do_transform:
                spatial = tuple(self.R.randint(0, k) for k in img.shape[1:])
                self.sampled_locs = [(i,) + spatial for i in range(img.shape[0])]
                if isinstance(intensity_range[0], Sequence):
                    self.sampled_k_intensity = [self.R.uniform(*p) for p in intensity_range]  # type: ignore
                else:
                    self.sampled_k_intensity = [self.R.uniform(*self.intensity_range)] * len(img)  # type: ignore

    def _make_sequence(self, x: np.ndarray) -> Sequence[Sequence[float]]:
        """
        Formats the sequence of intensities ranges to Sequence[Sequence[float]].
        """
        if self.intensity_range is not None:
            if not isinstance(self.intensity_range[0], Sequence):
                intensity_range = (ensure_tuple(self.intensity_range),) * x.shape[0]
                return intensity_range
            else:
                return ensure_tuple(self.intensity_range)
        else:
            # set default range if one not provided
            return self._set_default_range(x)

    def _set_default_range(self, x: np.ndarray) -> Sequence[Sequence[float]]:
        """
        Sets default intensity ranges to be sampled.

        Args:
            x (np.ndarray): tensor to fourier transform.
        """
        n_dims = len(x.shape[1:])

        k = np.fft.fftshift(np.fft.fftn(x, axes=tuple(range(-n_dims, 0))), axes=tuple(range(-n_dims, 0)))
        log_abs = np.log(np.absolute(k) + 1e-10)
        shifted_means = np.mean(log_abs, axis=tuple(range(-n_dims, 0))) * 2.5
        intensity_sequence = tuple((i * 0.95, i * 1.1) for i in shifted_means)
        return intensity_sequence

    def _to_numpy(self, img: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, torch.device]:
        if isinstance(img, torch.Tensor):
            return img.cpu().detach().numpy(), img.device
        else:
            return img, torch.device("cpu")
