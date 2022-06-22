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
from typing import Optional

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

import monai


def convert_to_tensor_complex(data: ndarray) -> Tensor:
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    inputs:
        data (np.array): Input numpy array
    outputs:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return monai.utils.type_conversion.convert_to_tensor(data)


def complex_abs(x: ndarray) -> ndarray:
    """
    Compute the absolute value of a complex array.
    inputs:
        x (np.array): Input numpy array with 2 channels in the last
        dimension representing real and imaginary parts.
    outputs:
        np.array: Absolute value along the last dimention
    """
    assert x.shape[-1] == 2
    return np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)


# mask functions
class MaskFunc:
    def __init__(self, center_fractions: list, accelerations: list) -> None:
        """
        inputs:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError("Number of center fractions should match number of accelerations")

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]
        return center_fraction, acceleration


def create_mask_for_mask_type(mask_type_str: str, center_fractions: list, accelerations: list) -> MaskFunc:
    """
    Create an under-sampling mask generator
    inputs:
        mask_type_str (string): denotes the mask type ('random','equispaced')
        center_fractions (List[float]): Fraction of low-frequency columns to be retained.
        If multiple values are provided, then one of these numbers is chosen uniformly each time.
        accelerations (List[int]): Amount of under-sampling. This should have the same length as center_fractions.
        If multiple values are provided, then one of these is chosen uniformly each time.
    outputs:
        callable mask function
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    else:
        raise Exception(f"{mask_type_str} not supported")


class RandomMaskFunc(MaskFunc):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the RandomMaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions: list, accelerations: list) -> None:
        """
        inputs:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.

            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError("Number of center fractions should match number of accelerations")

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, spatial_size: tuple, seed: Optional[int] = None) -> Tensor:
        """
        inputs:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        outputs:
            torch.Tensor: A mask of the specified shape.
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
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


class EquispacedMaskFunc(MaskFunc):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected with equal spacing at a proportion that reaches the
           desired acceleration rate taking into consideration the number of low frequencies. This
           ensures that the expected number of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the EquispacedMaskFunc
    object is called.
    """

    def __call__(self, spatial_size: tuple, seed: Optional[int] = None) -> Tensor:
        """
        inputs:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        outputs:
            torch.Tensor: A mask of the specified shape.
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
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


def apply_mask(
    data: Tensor, mask_func: Optional[MaskFunc] = None, mask: Optional[torch.Tensor] = None, seed: Optional[int] = None
) -> Tensor:
    """
    Subsample given k-space by multiplying with a mask.
    inputs:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
    outputs:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    if mask is None:
        mask = mask_func(shape, seed)
    return data * mask, mask
