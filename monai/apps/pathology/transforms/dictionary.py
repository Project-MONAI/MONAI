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
A collection of dictionary-based wrappers around the pathology transforms
defined in :py:class:`monai.apps.pathology.transforms.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import TYPE_CHECKING, Dict, Hashable, Mapping

from monai.apps.pathology.transforms.array import ExtractHEStains, NormalizeStainsMacenko
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    import cupy as cp
else:
    cp, _ = optional_import("cupy", "8.6.0", exact_version)


class ExtractHEStainsd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.ExtractHEStains`.
    Class to extract a target stain from an image, using the Macenko method for stain deconvolution.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        tli: transmitted light intensity. Defaults to 240.
        alpha: tolerance in percentile for the pseudo-min (alpha percentile)
            and pseudo-max (100 - alpha percentile). Defaults to 1.
        beta: absorbance threshold for transparent pixels. Defaults to 0.15
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to None.
        allow_missing_keys: don't raise exception if key is missing.

    Note:
        For more information refer to:
        - the original paper: Macenko et al., 2009 http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
        - the previous implementations:
            - MATLAB: https://github.com/mitkovetta/staining-normalization
            - Python: https://github.com/schaugf/HEnorm_python
    """

    def __init__(
        self,
        keys: KeysCollection,
        tli: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        max_cref: cp.ndarray = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.extractor = ExtractHEStains(tli=tli, alpha=alpha, beta=beta, max_cref=max_cref)

    def __call__(self, data: Mapping[Hashable, cp.ndarray]) -> Dict[Hashable, cp.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.extractor(d[key])
        return d


class NormalizeStainsMacenkod(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.NormalizeStainsMacenko`.

    Class to normalize patches/images to a reference or target image stain, using the Macenko method.

    Performs stain deconvolution of the source image using the ExtractHEStains
    class, to obtain the stain matrix and calculate the stain concentration matrix
    for the image. Then, performs the inverse Beer-Lambert transform to recreate the
    patch using the target H&E stain matrix provided. If no target stain provided, a default
    reference stain is used. Similarly, if no maximum stain concentrations are provided, a
    reference maximum stain concentrations matrix is used.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        tli: transmitted light intensity. Defaults to 240.
        alpha: tolerance in percentile for the pseudo-min (alpha percentile) and
            pseudo-max (100 - alpha percentile). Defaults to 1.
        beta: absorbance threshold for transparent pixels. Defaults to 0.15.
        target_he: target stain matrix. Defaults to None.
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to None.
        allow_missing_keys: don't raise exception if key is missing.

    Note:
        For more information refer to:
        - the original paper: Macenko et al., 2009 http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
        - the previous implementations:
            - MATLAB: https://github.com/mitkovetta/staining-normalization
            - Python: https://github.com/schaugf/HEnorm_python
    """

    def __init__(
        self,
        keys: KeysCollection,
        tli: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        target_he: cp.ndarray = None,
        max_cref: cp.ndarray = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = NormalizeStainsMacenko(
            tli=tli, alpha=alpha, beta=beta, target_he=target_he, max_cref=max_cref
        )

    def __call__(self, data: Mapping[Hashable, cp.ndarray]) -> Dict[Hashable, cp.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key])
        return d
