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
from numpy import ndarray
from torch import Tensor

from monai.apps.reconstruction.array import EquispacedKspaceMask, RandomKspaceMask
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import SpatialCrop
from monai.transforms.croppad.dictionary import Cropd
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import FastMRIKeys
from monai.utils.type_conversion import convert_to_tensor


class RandomKspaceMaskd(RandomizableTransform, MapTransform):
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
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.masker = RandomKspaceMask(
            center_fractions=center_fractions,
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandomKspaceMaskd":
        super().set_random_state(seed, state)
        self.masker.set_random_state(seed, state)
        return self

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
            d["target"] = d[key + meta][FastMRIKeys.RECON]  # type: ignore
            d["filename"] = d[key + meta][FastMRIKeys.FILENAME]  # type: ignore
        return d  # type: ignore


class EquispacedKspaceMaskd(RandomizableTransform, MapTransform):
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
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.masker = EquispacedKspaceMask(
            center_fractions=center_fractions, accelerations=accelerations, is_complex=is_complex
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "EquispacedKspaceMaskd":
        super().set_random_state(seed, state)
        self.masker.set_random_state(seed, state)
        return self

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
            d["target"] = d[key + meta][FastMRIKeys.RECON]  # type: ignore
            d["filename"] = d[key + meta][FastMRIKeys.FILENAME]  # type: ignore
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
            roi_center = tuple(i // 2 for i in image.shape[1:])
            cropper = SpatialCrop(roi_center=roi_center, roi_size=roi_size)
            d[key] = convert_to_tensor(cropper(d[key]))
        return d


class InputTargetNormalizeIntensityd(MapTransform):
    """
    Dictionary-based wrapper of
    :py:class:`monai.transforms.NormalizeIntensity`.
    This is similar to :py:class:`monai.transforms.NormalizeIntensityd`
    but is different since (1) it automatically normalizes the target (see
    "gt_key" in the parameters below for a definition for target) based
    on the input data, and (2) it also returns mean and std after normalization. The
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
        gt_key: denotes the target to be normalized based on input
            statistics. It is typically set to "target" which denotes
            the ground-truth data.
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
        gt_key: str = "target",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.default_normalizer = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise, dtype)
        self.gt_key = gt_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        """
        This will normalize the input first, and then according to its mean-std,
        it will normalize the target. The target is denoted by self.gt_key whose default
        is "target" denoting the ground-truth data.

        The first dimension of input is reserved for channel (e.g., could be num_slices).
        The rest of the dimensions should be spatial. So for example, the shape is (C,H,W,D)
        for 3D samples.

        Args:
            data: is a dictionary containing (key,value) pairs from
                the loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            if self.default_normalizer.channel_wise:
                # perform channel-wise normalization
                # compute mean of each channel (or slice) in the input for mean-std normalization
                # subtrahend will have the same shape as input, for example (C,W,D) for a 2D data (or a pseudo-3D data)
                if self.default_normalizer.subtrahend is None:
                    subtrahend = np.array(
                        [val.mean() if isinstance(val, ndarray) else val.float().mean().item() for val in d[key]]
                    )
                # users can define default values instead of mean
                else:
                    subtrahend = self.default_normalizer.subtrahend  # type: ignore

                # compute std of each channel (or slice) in the input for mean-std normalization
                # will have the same shape as subtrahend
                if self.default_normalizer.divisor is None:
                    divisor = np.array(
                        [
                            val.std() if isinstance(val, ndarray) else val.float().std(unbiased=False).item()
                            for val in d[key]
                        ]
                    )
                else:
                    # users can define default values instead of std
                    divisor = self.default_normalizer.divisor  # type: ignore
            else:
                # perform ordinary normalization (not channel-wise)
                # subtrahend will be a scalar and is the mean of d[key], unless user specifies another value
                if self.default_normalizer.subtrahend is None:
                    subtrahend = d[key].mean() if isinstance(d[key], ndarray) else d[key].float().mean().item()  # type: ignore
                # users can define default values instead of mean
                else:
                    subtrahend = self.default_normalizer.subtrahend  # type: ignore

                # divisor will be a scalar and is the std of d[key], unless user specifies another value
                if self.default_normalizer.divisor is None:
                    if isinstance(d[key], ndarray):
                        divisor = d[key].std()  # type: ignore
                    else:
                        divisor = d[key].float().std(unbiased=False).item()  # type: ignore
                else:
                    # users can define default values instead of std
                    divisor = self.default_normalizer.divisor  # type: ignore

            # this creates a new normalizer instance for each sample
            normalizer = NormalizeIntensity(
                subtrahend,
                divisor,
                self.default_normalizer.nonzero,
                self.default_normalizer.channel_wise,
                self.default_normalizer.dtype,
            )
            d[key] = normalizer(d[key])
            d[self.gt_key] = normalizer(d[self.gt_key])  # target is normalized according to input
            d["mean"] = subtrahend
            d["std"] = divisor
        return d
