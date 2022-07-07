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


from typing import Dict, Hashable, Mapping, Optional, Sequence

import numpy as np
from my_mri_array import EquispacedKspaceMask, RandomKspaceMask
from numpy import ndarray
from torch import Tensor

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import SpatialCrop
from monai.transforms.croppad.dictionary import Cropd
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.transform import MapTransform
from monai.utils.type_conversion import convert_to_tensor


class RandomKspaceMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandomKspaceMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        center_fractions: Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is
            chosen uniformly each time.
        accelerations: Amount of under-sampling. This should have the
            same length as center_fractions. If multiple values are provided,
            then one of these is chosen uniformly each time.
        spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data; it's
            also 2 for psuedo-3D datasets like the fastMRI dataset).
            The last spatial dim is selected for sampling. For the fastMRI
            dataset, k-space has the form (...,num_slices,num_coils,H,W)
            and sampling is done along W. For a general 3D data with the
            shape (...,num_coils,H,W,D), sampling is done along D.
        is_complex: if True, then the last dimension will be reserved
            for real/imaginary parts.
        seed: Seed for the random number generator. Setting the seed
            ensures the same mask is generated each time for the same shape.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandomKspaceMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        spatial_dims: int = 2,
        is_complex: bool = True,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.masker = RandomKspaceMask(
            center_fractions=center_fractions,
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
            seed=seed,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):  # key is typically just "kspace"
            d[key + "_masked"], d[key + "_masked_ifft"] = self.masker(d[key])
            d["mask"] = self.masker.mask  # type: ignore
            meta = "_meta_dict"
            d["target"] = d[key + meta]["reconstruction_rss"]  # type: ignore
            d["filename"] = d[key + meta]["filename"]  # type: ignore
        return d  # type: ignore


class EquispacedKspaceMaskd(MapTransform):
    """
    Dictionary-based wrapper of
    :py:class:`monai.transforms.EquispacedKspaceMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        center_fractions: Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is
            chosen uniformly each time.
        accelerations: Amount of under-sampling. This should have the same
            length as center_fractions. If multiple values are provided,
            then one of these is chosen uniformly each time.
        spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data;
            it's also 2 for  psuedo-3D datasets like the fastMRI dataset).
            The last spatial dim is selected for sampling. For the fastMRI
            dataset, k-space has the form (...,num_slices,num_coils,H,W)
            and sampling is done along W. For a general 3D data with the shape
            (...,num_coils,H,W,D), sampling is done along D.
        is_complex: if True, then the last dimension will be reserved
            for real/imaginary parts.
        seed: Seed for the random number generator. Setting the seed
            ensures the same mask is generated each time for the same shape.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = EquispacedKspaceMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        spatial_dims: int = 2,
        is_complex: bool = True,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.masker = EquispacedKspaceMask(
            center_fractions=center_fractions, accelerations=accelerations, is_complex=is_complex, seed=seed
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from
            the loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):  # key is typically just "kspace"
            d[key + "_masked"], d[key + "_masked_ifft"] = self.masker(d[key])
            d["mask"] = self.masker.mask  # type: ignore
            meta = "_meta_dict"
            d["target"] = d[key + meta]["reconstruction_rss"]  # type: ignore
            d["filename"] = d[key + meta]["filename"]  # type: ignore
        return d  # type: ignore


class TargetBasedSpatialCropd(Cropd):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialCrop`.
    This is similar to :py:class:`monai.transforms.SpatialCropd` but is
    different since it crops the data based on the "Target" spatial size.
    So for every new sample, the roi_center and roi_size will be different.
    The first dimension is reserved for channel (e.g., could be num_slices).
    The rest of the dimensions should be spatial. So the shape is (C,H,W,D)
    for 3D samples.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:

        super().__init__(keys, cropper=None, allow_missing_keys=allow_missing_keys)  # type: ignore

    def __call__(self, data: Mapping[Hashable, Tensor]) -> Dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from
            the loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            # key is typically just "kspace_masked_ifft" for fastMRI data
            image = d[key]
            roi_size = d["target"].shape[1:]
            roi_center = tuple([i // 2 for i in image.shape[1:]])
            cropper = SpatialCrop(roi_center=roi_center, roi_size=roi_size)
            d[key] = convert_to_tensor(cropper(d[key]))
        return d


class DetailedNormalizeIntensityd(MapTransform):
    """
    Dictionary-based wrapper of
    :py:class:`monai.transforms.NormalizeIntensity`.
    This is similar to :py:class:`monai.transforms.NormalizeIntensityd`
    but is different since (1) it automatically normalizes the target based
    on the data, and (2) it also returns mean and std after normalization. The
    first dimension is reserved for channel (e.g., could be num_slices).
    The rest of the dimensions should be spatial. So the shape is (C,H,W,D)
    for 3D samples.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately,
            otherwise, calculate on the entire image directly. default
            to False.
        dtype: output data type, if None, same as input image. defaults
            to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = NormalizeIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        subtrahend: Optional[NdarrayOrTensor] = None,
        divisor: Optional[NdarrayOrTensor] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.default_normalizer = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise, dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from
            the loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            if self.default_normalizer.subtrahend is None:
                subtrahend = np.array([val.mean() if isinstance(val, ndarray) else val.mean().item() for val in d[key]])
            else:
                subtrahend = self.default_normalizer.subtrahend  # type: ignore

            if self.default_normalizer.divisor is None:
                divisor = np.array([val.std() if isinstance(val, ndarray) else val.std().item() for val in d[key]])
            else:
                divisor = self.default_normalizer.divisor  # type: ignore

            normalizer = NormalizeIntensity(
                subtrahend,
                divisor,
                self.default_normalizer.nonzero,
                self.default_normalizer.channel_wise,
                self.default_normalizer.dtype,
            )
            d[key] = normalizer(d[key])
            d["target"] = normalizer(d["target"])
            d["mean"] = subtrahend
            d["std"] = divisor
        return d
