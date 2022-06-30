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

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from torch import Tensor

from monai.transforms.transform import Randomizable, Transform
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_to_tensor


class KspaceMask(Randomizable, ABC):
    """
    A basic class for under-sampling mask setup. It provides common features for under-sampling mask generators.
    For example, RandomMaskFunc and EquispacedMaskFunc (two mask transform objects defined right after this module)
    both inherit MaskFunc to properly setup properties like the acceleration factor.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        spatial_size: Sequence[int],
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations: Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time.
            spatial_size: The shape of the mask to be created. The shape should have
                at least 3 dimensions (H,W,2). Samples are drawn along the second last dimension (W).
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError("Number of center fractions should match number of accelerations")

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.spatial_size = spatial_size
        self.R.seed(seed)

    @abstractmethod
    def __call__(self, kspace: Tensor):
        """
        This is an extra instance to allow for defining new mask generators.
        For creating other mask transforms, define a new class and simply overwrite __call__.
        See an example of this in 'RandomMaskFunc.'

        Args:
            kspace: The input k-space data. This should have at least 3 dimensions (...,H,W,2), where
                dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
                2 (for complex values).
        """
        raise NotImplementedError

    def randomize_choose_acceleration(self) -> Sequence[float]:
        """
        If multiple values are provided for center_fractions and accelerations, this function selects one value
        uniformly for each training/test sample.

        Returns:
            A tuple containing
                (1) center_fraction: chosen fraction of center kspace lines to exclude from under-sampling
                (2) acceleration: chosen acceleration factor
        """
        choice = self.R.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]
        return center_fraction, acceleration


class RandomKspaceMask(KspaceMask, Transform):
    """
    The mask uniformly selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
    1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
    low-frequencies
    2. The other columns are selected uniformly at random with a probability equal to:
    prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the RandomMaskFunc object is
    called.

    Example:
        If accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
        is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
        probability that 8-fold acceleration with 4% center fraction is selected.

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    """

    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: Tensor) -> Tensor:
        """
        Args:
            kspace: The input k-space data. This should have at least 3 dimensions (...,H,W,2), where
                dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
                2 (for complex values).

        Returns:
            The under-sampled kspace
        """
        num_cols = self.spatial_size[-2]
        center_fraction, acceleration = self.randomize_choose_acceleration()

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.R.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in self.spatial_size]
        mask_shape[-2] = num_cols

        mask = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        # under-sample the ksapce
        masked = mask * kspace
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask
        return masked_kspace


class EquispacedKspaceMask(KspaceMask, Transform):
    """
    The mask selects an equi-distant subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
    1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
    low-frequencies
    2. The other columns are selected with equal spacing at a proportion that reaches the
    desired acceleration rate taking into consideration the number of low frequencies. This
    ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the EquispacedMaskFunc
    object is called.

    Example:
        If accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
        is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
        probability that 8-fold acceleration with 4% center fraction is selected.

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    """

    backend = [TransformBackends.TORCH]

    def __call__(self, kspace: Tensor) -> Tensor:
        """
        Args:
            kspace: The input k-space data. This should have at least 3 dimensions (...,H,W,2), where
                dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
                2 (for complex values).

        Returns:
            The under-sampled kspace
        """
        num_cols = self.spatial_size[-2]
        center_fraction, acceleration = self.randomize_choose_acceleration()
        num_low_freqs = int(round(num_cols * center_fraction))

        # Create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
        offset = self.R.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True

        # Reshape the mask
        mask_shape = [1 for _ in self.spatial_size]
        mask_shape[-2] = num_cols
        mask = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        # under-sample the ksapce
        masked = mask * kspace
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask
        return masked_kspace
