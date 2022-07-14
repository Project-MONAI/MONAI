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


from abc import abstractmethod
from typing import Sequence

import numpy as np
from torch import Tensor

from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.fft_utils import ifftn_centered
from monai.transforms.transform import RandomizableTransform
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_to_tensor


class KspaceMask(RandomizableTransform):
    """
    A basic class for under-sampling mask setup. It provides common
    features for under-sampling mask generators.
    For example, RandomMaskFunc and EquispacedMaskFunc (two mask
    transform objects defined right after this module)
    both inherit MaskFunc to properly setup properties like the
    acceleration factor.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        spatial_dims: int = 2,
        is_complex: bool = True,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers
                is chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the
                same length as center_fractions. If multiple values are
                provided, then one of these is chosen uniformly each time.
            spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data;
                it's also 2 for psuedo-3D datasets like the fastMRI dataset).
                The last spatial dim is selected for sampling. For the fastMRI
                dataset, k-space has the form (...,num_slices,num_coils,H,W)
                and sampling is done along W. For a general 3D data with the
                shape (...,num_coils,H,W,D), sampling is done along D.
            is_complex: if True, then the last dimension will be reserved for
                real/imaginary parts.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError(
                "Number of center fractions \
                should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.spatial_dims = spatial_dims
        self.is_complex = is_complex

    @abstractmethod
    def __call__(self, kspace: NdarrayOrTensor):
        """
        This is an extra instance to allow for defining new mask generators.
        For creating other mask transforms, define a new class and simply
        override __call__. See an example of this in
        :py:class:`monai.apps.reconstruction.transforms.array.RandomKspacemask`.

        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data.
        """
        raise NotImplementedError

    def randomize_choose_acceleration(self) -> Sequence[float]:
        """
        If multiple values are provided for center_fractions and
        accelerations, this function selects one value uniformly
        for each training/test sample.

        Returns:
            A tuple containing
                (1) center_fraction: chosen fraction of center kspace
                lines to exclude from under-sampling
                (2) acceleration: chosen acceleration factor
        """
        choice = self.R.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]
        return center_fraction, acceleration


class RandomKspaceMask(KspaceMask):
    """
    This k-space mask transform under-samples the k-space according to a
    random sampling pattern. Precisely, it uniformly selects a subset of
    columns from the input k-space data. If the k-space data has N columns,
    the mask picks out:

    1. N_low_freqs = (N * center_fraction) columns in the center
    corresponding to low-frequencies

    2. The other columns are selected uniformly at random with a probability
    equal to:
    prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to
    (N / acceleration)

    It is possible to use multiple center_fractions and accelerations,
    in which case one possible (center_fraction, acceleration) is chosen
    uniformly at random each time the transform is called.

    Example:
        If accelerations = [4, 8] and center_fractions = [0.08, 0.04],
        then there is a 50% probability that 4-fold acceleration with 8%
        center fraction is selected and a 50% probability that 8-fold
        acceleration with 4% center fraction is selected.

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    """

    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        """
        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data. The last spatial dim is selected for sampling. For the
                fastMRI dataset, k-space has the form
                (...,num_slices,num_coils,H,W) and sampling is done along W.
                For a general 3D data with the shape (...,num_coils,H,W,D),
                sampling is done along D.

        Returns:
            A tuple containing
                (1) the under-sampled kspace
                (2) absolute value of the inverse fourier of the under-sampled kspace
        """
        kspace_t = convert_to_tensor_complex(kspace)
        spatial_size = kspace_t.shape
        num_cols = spatial_size[-1]
        if self.is_complex:  # for complex data
            num_cols = spatial_size[-2]

        center_fraction, acceleration = self.randomize_choose_acceleration()

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.R.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in spatial_size]
        if self.is_complex:
            mask_shape[-2] = num_cols
        else:
            mask_shape[-1] = num_cols

        mask = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        # under-sample the ksapce
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # compute inverse fourier of the masked kspace
        masked_kspace_ifft: Tensor = convert_to_tensor(
            complex_abs(ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex))
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(masked_kspace_ifft, spatial_dim=-self.spatial_dims - 1)
        )
        return masked_kspace, masked_kspace_ifft_rss


class EquispacedKspaceMask(KspaceMask):
    """
    This k-space mask transform under-samples the k-space according to an
    equi-distant sampling pattern. Precisely, it selects an equi-distant
    subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:

    1. N_low_freqs = (N * center_fraction) columns in the center corresponding
    to low-frequencies

    2. The other columns are selected with equal spacing at a proportion that
    reaches the desired acceleration rate taking into consideration the number
    of low frequencies. This ensures that the expected number of columns
    selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in
    which case one possible (center_fraction, acceleration) is chosen
    uniformly at random each time the EquispacedMaskFunc object is called.

    Example:
        If accelerations = [4, 8] and center_fractions = [0.08, 0.04],
        then there is a 50% probability that 4-fold acceleration with 8%
        center fraction is selected and a 50% probability that 8-fold
        acceleration with 4% center fraction is selected.

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    """

    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        """
        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data. The last spatial dim is selected for sampling. For the
                fastMRI multi-coil dataset, k-space has the form
                (...,num_slices,num_coils,H,W) and sampling is done along W.
                For a general 3D data with the shape (...,num_coils,H,W,D),
                sampling is done along D.

        Returns:
            A tuple containing
                (1) the under-sampled kspace
                (2) absolute value of the inverse fourier of the under-sampled kspace
        """
        kspace_t = convert_to_tensor_complex(kspace)
        spatial_size = kspace_t.shape
        num_cols = spatial_size[-1]
        if self.is_complex:  # for complex data
            num_cols = spatial_size[-2]

        center_fraction, acceleration = self.randomize_choose_acceleration()
        num_low_freqs = int(round(num_cols * center_fraction))

        # Create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Determine acceleration rate by adjusting for the
        # number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
        offset = self.R.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True

        # Reshape the mask
        mask_shape = [1 for _ in spatial_size]
        if self.is_complex:
            mask_shape[-2] = num_cols
        else:
            mask_shape[-1] = num_cols

        mask = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        # under-sample the ksapce
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # compute inverse fourier of the masked kspace
        masked_kspace_ifft: Tensor = convert_to_tensor(
            complex_abs(ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex))
        )
        # combine coil images (it is assumed that the coil dimension is
        # the first dimension before spatial dimensions)
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(masked_kspace_ifft, spatial_dim=-self.spatial_dims - 1)
        )
        return masked_kspace, masked_kspace_ifft_rss
