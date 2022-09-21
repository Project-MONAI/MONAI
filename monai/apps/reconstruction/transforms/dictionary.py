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

from monai.apps.reconstruction.transforms.array import EquispacedKspaceMask, RandomKspaceMask
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import SpatialCrop
from monai.transforms.croppad.dictionary import Cropd
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import FastMRIKeys
from monai.utils.type_conversion import convert_to_tensor


class ExtractDataKeyFromMetaKeyd(MapTransform):
    """
    Moves keys from meta to data. It is useful when a dataset of paired samples
    is loaded and certain keys should be moved from meta to data.

    Args:
        keys: keys to be transferred from meta to data
        meta_key: the meta key where all the meta-data is stored
        allow_missing_keys: don't raise exception if key is missing

    Example:
        When the fastMRI dataset is loaded, "kspace" is stored in the data dictionary,
        but the ground-truth image with the key "reconstruction_rss" is stored in the meta data.
        In this case, ExtractDataKeyFromMetaKeyd moves "reconstruction_rss" to data.
    """

    def __init__(self, keys: KeysCollection, meta_key: str, allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.meta_key = meta_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.keys:
            if key in d[self.meta_key]:
                d[key] = d[self.meta_key][key]  # type: ignore
            elif not self.allow_missing_keys:
                raise KeyError(
                    f"Key `{key}` of transform `{self.__class__.__name__}` was missing in the meta data"
                    " and allow_missing_keys==False."
                )
        return d  # type: ignore


class RandomKspaceMaskd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.reconstruction.transforms.array.RandomKspacemask`.
    Other mask transforms can inherit from this class, for example:
    :py:class:`monai.apps.reconstruction.transforms.dictionary.EquispacedKspaceMaskd`.

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
        for key in self.key_iterator(d):
            d[key + "_masked"], d[key + "_masked_ifft"] = self.masker(d[key])
            d[FastMRIKeys.MASK] = self.masker.mask

        return d  # type: ignore


class EquispacedKspaceMaskd(RandomKspaceMaskd):
    """
    Dictionary-based wrapper of
    :py:class:`monai.apps.reconstruction.transforms.array.EquispacedKspaceMask`.

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
        self.masker = EquispacedKspaceMask(  # type: ignore
            center_fractions=center_fractions,
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "EquispacedKspaceMaskd":
        super().set_random_state(seed, state)
        self.masker.set_random_state(seed, state)
        return self


class ReferenceBasedSpatialCropd(Cropd):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialCrop`.
    This is similar to :py:class:`monai.transforms.SpatialCropd` which is a
    general purpose cropper to produce sub-volume region of interest (ROI).
    Their difference is that this transform does cropping according to a reference image.

    If a dimension of the expected ROI size is larger than the input image size, will not crop that dimension.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        ref_key: key of the item to be used to crop items of "keys"
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        In an image reconstruction task, let keys=["image"] and ref_key=["target"].
        Also, let data be the data dictionary. Then, ReferenceBasedSpatialCropd
        center-crops data["image"] based on the spatial size of data["target"] by
        calling :py:class:`monai.transforms.SpatialCrop`.
    """

    def __init__(self, keys: KeysCollection, ref_key: str, allow_missing_keys: bool = False) -> None:

        super().__init__(keys, cropper=None, allow_missing_keys=allow_missing_keys)  # type: ignore
        self.ref_key = ref_key

    def __call__(self, data: Mapping[Hashable, Tensor]) -> Dict[Hashable, Tensor]:
        """
        This transform can support to crop ND spatial (channel-first) data.
        It also supports pseudo ND spatial data (e.g., (C,H,W) is a pseudo-3D
        data point where C is the number of slices)

        Args:
            data: is a dictionary containing (key,value) pairs from
                the loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)

        # compute roi_size according to self.ref_key
        roi_size = d[self.ref_key].shape[1:]  # first dimension is not spatial (could be channel)

        # crop keys
        for key in self.key_iterator(d):
            image = d[key]
            roi_center = tuple(i // 2 for i in image.shape[1:])
            cropper = SpatialCrop(roi_center=roi_center, roi_size=roi_size)
            d[key] = convert_to_tensor(cropper(d[key]))
        return d


class ReferenceBasedNormalizeIntensityd(MapTransform):
    """
    Dictionary-based wrapper of
    :py:class:`monai.transforms.NormalizeIntensity`.
    This is similar to :py:class:`monai.transforms.NormalizeIntensityd`
    and can normalize non-zero values or the entire image. The difference
    is that this transform does normalization according to a reference image.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        ref_key: key of the item to be used to normalize items of "keys"
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately,
            otherwise, calculate on the entire image directly. default
            to False.
        dtype: output data type, if None, same as input image. defaults
            to float32.
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        In an image reconstruction task, let keys=["image", "target"] and ref_key=["image"].
        Also, let data be the data dictionary. Then, ReferenceBasedNormalizeIntensityd
        normalizes data["target"] and data["image"] based on the mean-std of data["image"] by
        calling :py:class:`monai.transforms.NormalizeIntensity`.
    """

    backend = NormalizeIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        ref_key: str,
        subtrahend: Optional[NdarrayOrTensor] = None,
        divisor: Optional[NdarrayOrTensor] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.default_normalizer = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise, dtype)
        self.ref_key = ref_key

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        """
        This transform can support to normalize ND spatial (channel-first) data.
        It also supports pseudo ND spatial data (e.g., (C,H,W) is a pseudo-3D
        data point where C is the number of slices)

        Args:
            data: is a dictionary containing (key,value) pairs from
                the loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)

        # prepare the normalizer based on self.ref_key
        if self.default_normalizer.channel_wise:
            # perform channel-wise normalization
            # compute mean of each channel in the input for mean-std normalization
            # subtrahend will have the same shape as image, for example (C,W,D) for a 2D data
            if self.default_normalizer.subtrahend is None:
                subtrahend = np.array(
                    [val.mean() if isinstance(val, ndarray) else val.float().mean().item() for val in d[self.ref_key]]
                )
            # users can define default values instead of mean
            else:
                subtrahend = self.default_normalizer.subtrahend  # type: ignore

            # compute std of each channel in the input for mean-std normalization
            # will have the same shape as subtrahend
            if self.default_normalizer.divisor is None:
                divisor = np.array(
                    [
                        val.std() if isinstance(val, ndarray) else val.float().std(unbiased=False).item()
                        for val in d[self.ref_key]
                    ]
                )
            else:
                # users can define default values instead of std
                divisor = self.default_normalizer.divisor  # type: ignore
        else:
            # perform ordinary normalization (not channel-wise)
            # subtrahend will be a scalar and is the mean of d[self.ref_key], unless user specifies another value
            if self.default_normalizer.subtrahend is None:
                if isinstance(d[self.ref_key], ndarray):
                    subtrahend = d[self.ref_key].mean()  # type: ignore
                else:
                    subtrahend = d[self.ref_key].float().mean().item()  # type: ignore
            # users can define default values instead of mean
            else:
                subtrahend = self.default_normalizer.subtrahend  # type: ignore

            # divisor will be a scalar and is the std of d[self.ref_key], unless user specifies another value
            if self.default_normalizer.divisor is None:
                if isinstance(d[self.ref_key], ndarray):
                    divisor = d[self.ref_key].std()  # type: ignore
                else:
                    divisor = d[self.ref_key].float().std(unbiased=False).item()  # type: ignore
            else:
                # users can define default values instead of std
                divisor = self.default_normalizer.divisor  # type: ignore

        # this creates a new normalizer instance based on self.ref_key
        normalizer = NormalizeIntensity(
            subtrahend,
            divisor,
            self.default_normalizer.nonzero,
            self.default_normalizer.channel_wise,
            self.default_normalizer.dtype,
        )

        # save mean and std
        d["mean"] = subtrahend
        d["std"] = divisor

        # perform normalization
        for key in self.key_iterator(d):
            d[key] = normalizer(d[key])

        return d
