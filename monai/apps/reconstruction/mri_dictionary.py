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

from mri_array import EquispacedKspaceMask, RandomKspaceMask
from torch import Tensor

from monai.config import KeysCollection
from monai.transforms.transform import MapTransform


class RandomKspaceMaskd(MapTransform):
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

    backend = RandomKspaceMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        spatial_size: Sequence[int],
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.masker = RandomKspaceMask(
            center_fractions=center_fractions, accelerations=accelerations, spatial_size=spatial_size, seed=seed
        )

    def __call__(self, data: Mapping[Hashable, Tensor]) -> Dict[Hashable, Tensor]:
        """
        Args:
            kspace: The input k-space data. This should have at least 3 dimensions (...,H,W,2), where
                dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
                2 (for complex values).

        Returns:
            The under-sampled kspace
        """
        d = dict(data)
        for key in self.key_iterator(d):  # key is typically just "kspace"
            d[key] = self.masker(d[key])
        return d


class EquispacedKspaceMaskd(MapTransform):
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

    backend = EquispacedKspaceMask.backend

    def __inint__(
        self,
        keys: KeysCollection,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        spatial_size: Sequence[int],
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.masker = EquispacedKspaceMask(
            center_fractions=center_fractions, accelerations=accelerations, spatial_size=spatial_size, seed=seed
        )

    def __call__(self, data: Mapping[Hashable, Tensor]) -> Dict[Hashable, Tensor]:
        """
        Args:
            kspace: The input k-space data. This should have at least 3 dimensions (...,H,W,2), where
                dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
                2 (for complex values).

        Returns:
            The under-sampled kspace
        """
        d = dict(data)
        for key in self.key_iterator(d):  # key is typically just "kspace"
            d[key] = self.masker(d[key])
        return d
