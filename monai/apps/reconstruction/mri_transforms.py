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

from typing import Optional, Sequence

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.type_conversion import convert_data_type, convert_to_tensor


def convert_to_tensor_complex(data: ndarray) -> Tensor:
    """
    Convert numpy array to PyTorch tensor.
    For complex arrays, the real and imaginary
    parts are stacked along the last dimension.

    Args:
        data: Input numpy array

    Returns:
        PyTorch version of the data

    Example:
        .. code-block:: python

            import numpy as np
            data = np.array([ [1+1j, 1-1j], [2+2j, 2-2j] ])
            # the following line prints (2,2)
            print(data.shape)
            # the following line prints torch.Size([2, 2, 2])
            print(convert_to_tensor_complex(data).shape)
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return convert_data_type(data, torch.Tensor)[0]


def complex_abs(x: NdarrayOrTensor) -> NdarrayOrTensor:
    """
    Compute the absolute value of a complex array.

    Args:
        x: Input array/tensor with 2 channels in the last dimension representing real and imaginary parts.

    Returns:
        Absolute value along the last dimention

    Example:
        .. code-block:: python

            import numpy as np
            x = np.array([3,4])[np.newaxis]
            # the following line prints 5
            print(complex_abs(x))
    """
    assert x.shape[-1] == 2
    return (x[..., 0] ** 2 + x[..., 1] ** 2) ** 0.5


class MaskFunc:
    """
    A basic class for under-sampling mask setup. It provides common features for under-sampling msak generators.
    For example, RandomMaskFunc and EquispacedMaskFunc (two mask-generating objects defined right after this module)
    both inherit MaskFunc to properly setup properties like the acceleration factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[float]):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations: Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError("Number of center fractions should match number of accelerations")

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, spatial_size: Sequence[int], seed: Optional[int] = None) -> Tensor:
        """
        This is an extra instance to allow for defining a default mask generator here.
        For creating other mask functions, define a new class and simply overwrite __call__.
        See an example of this in 'RandomMaskFunc.'

        Args:
            spatial_size: The shape of the mask to be created. The shape should have
                at least 3 dimensions (H,W,2). Samples are drawn along the second last dimension (W).
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.

        Returns:
            mask which is the under-sampling mask.
        """
        mask_shape = [1 for _ in spatial_size]
        mask_shape[-2] = spatial_size[-2]
        return torch.ones(mask_shape)

    def choose_acceleration(self) -> Sequence[float]:
        """
        If multiple values are provided for center_fractions and accelerations, this function selects one value
        uniformly for each training/test sample.

        Returns:
            A tuple containing
                (1) center_fraction: chosen fraction of center kspace lines to exclude from under-sampling
                (2) acceleration: chosen acceleration factor
        """
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]
        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    RandomMaskFunc creates a sub-sampling mask of a given spatial size.

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
    """

    def __call__(self, spatial_size: Sequence[int], seed: Optional[int] = None) -> Tensor:
        """
        Args:
            spatial_size: The shape of the mask to be created. The shape should have
                at least 3 dimensions (..,H,W,2). Samples are drawn along the second last dimension (W).
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.

        Returns:
            final_mask which is the under-sampling mask.
        """
        if len(spatial_size) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        self.rng.seed(seed)
        num_cols = spatial_size[-2]
        center_fraction, acceleration = self.choose_acceleration()

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in spatial_size]
        mask_shape[-2] = num_cols

        final_mask: Tensor = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        return final_mask


class EquispacedMaskFunc(MaskFunc):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

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
    """

    def __call__(self, spatial_size: Sequence[int], seed: Optional[int] = None) -> Tensor:
        """
        Args:
            spatial_size: The shape of the mask to be created. The shape should have
                at least 3 dimensions (H,W,2). Samples are drawn along the second last dimension (W).
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.

        Returns:
            final_mask which is the under-sampling mask.
        """
        if len(spatial_size) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        self.rng.seed(seed)
        center_fraction, acceleration = self.choose_acceleration()
        num_cols = spatial_size[-2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # Create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # Determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
        offset = self.rng.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True

        # Reshape the mask
        mask_shape = [1 for _ in spatial_size]
        mask_shape[-2] = num_cols
        final_mask: Tensor = convert_to_tensor(mask.reshape(*mask_shape).astype(np.float32))

        return final_mask


def apply_mask(
    data: Tensor, mask_func: MaskFunc, mask: Optional[Tensor] = None, seed: Optional[int] = None
) -> Sequence[Tensor]:
    """
    Subsample the given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions (...,H,W,2), where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func: A function that takes a shape and a random
            number seed and returns a mask.
        mask: the undersampling mask. when this is provided,
            mask_func will be ignored
        seed: Seed for the random number generator.

    Returns:
        tuple containing
            (1) masked data (torch.Tensor): Subsampled k-space data
            (2) mask (torch.Tensor): The generated mask
    """
    if mask is None:
        shape = np.array(data.shape)
        shape[:-3] = 1
        mask = mask_func(shape.tolist(), seed)
    return data * mask, mask


def create_mask_for_mask_type(
    mask_type_str: str, center_fractions: Sequence[float], accelerations: Sequence[float]
) -> MaskFunc:
    """
    Create an under-sampling mask generator

    Args:
        mask_type_str: denotes the mask type ('random','equispaced')
        center_fractions: Fraction of low-frequency columns to be retained (i.e., excluded from sampling).
            If multiple values are provided, then one of these numbers is chosen uniformly each time.
        accelerations: Amount of under-sampling. This should have the same length as center_fractions.
            If multiple values are provided, then one of these is chosen uniformly each time.

    Returns:
        callable mask function
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    else:
        raise Exception(f"{mask_type_str} not supported")
