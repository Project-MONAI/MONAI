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
A collection of dictionary-based wrappers around the pathology transforms
defined in :py:class:`monai.apps.pathology.transforms.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Dict, Hashable, Mapping, Union

import numpy as np

from monai.config import KeysCollection
from monai.transforms.transform import MapTransform

from .array import HEStainExtractor, StainNormalizer


class HEStainExtractord(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.HEStainExtractor`.
    Class to extract a target stain from an image, using stain deconvolution.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        source_intensity: transmitted light intensity.
            Defaults to 240.
        alpha: percentiles to ignore for outliers, so to calculate min and max,
            if only consider (alpha, 100-alpha) percentiles. Defaults to 1.
        beta: absorbance threshold for transparent pixels.
            Defaults to 0.15
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        source_intensity: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.extractor = HEStainExtractor(source_intensity=source_intensity, alpha=alpha, beta=beta)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.extractor(d[key])
        return d


class StainNormalizerd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.StainNormalizer`.

    Normalize images to a reference stain color matrix.

    First, it extracts the stain coefficient matrix from the image using the provided stain extractor.
    Then, it calculates the stain concentrations based on Beer-Lamber Law.
    Next, it reconstructs the image using the provided reference stain matrix (stain-normalized image).

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        source_intensity: transmitted light intensity.
            Defaults to 240.
        alpha: percentiles to ignore for outliers, so to calculate min and max,
            if only consider (alpha, 100-alpha) percentiles. Defaults to 1.
        ref_stain_coeff: reference stain attenuation coefficient matrix.
            Defaults to ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)).
        ref_max_conc: reference maximum stain concentrations for
            Hematoxylin & Eosin (H&E). Defaults to (1.9705, 1.0308).
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        source_intensity: float = 240,
        alpha: float = 1,
        ref_stain_coeff: Union[tuple, np.ndarray] = ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)),
        ref_max_conc: Union[tuple, np.ndarray] = (1.9705, 1.0308),
        stain_extractor=None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = StainNormalizer(
            source_intensity=source_intensity,
            alpha=alpha,
            ref_stain_coeff=ref_stain_coeff,
            ref_max_conc=ref_max_conc,
            stain_extractor=stain_extractor,
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key])
        return d


HEStainExtractorDict = HEStainExtractorD = HEStainExtractord
StainNormalizerDict = StainNormalizerD = StainNormalizerd
