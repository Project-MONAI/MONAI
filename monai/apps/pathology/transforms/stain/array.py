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

from typing import Union

import numpy as np

from monai.transforms.transform import Transform


class HEStainExtractor(Transform):
    """Extract stain coefficients from an image.

    Args:
        source_intensity: transmitted light intensity.
            Defaults to 240.
        alpha: percentiles to ignore for outliers, so to calculate min and max,
            if only consider (alpha, 100-alpha) percentiles. Defaults to 1.
        beta: absorbance threshold for transparent pixels.
            Defaults to 0.15

    Note:
        Please refer to this paper for further information on the method:
        Macenko et al., 2009 http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    """

    def __init__(self, source_intensity: float = 240, alpha: float = 1, beta: float = 0.15) -> None:
        self.source_intensity = source_intensity
        self.alpha = alpha
        self.beta = beta

    def calculate_flat_absorbance(self, image):
        """Calculate absorbace and remove transparent pixels"""
        # calculate absorbance
        image = image.astype(np.float32, copy=False) + 1.0
        absorbance = -np.log(image.clip(max=self.source_intensity) / self.source_intensity)

        # reshape to form a CxN matrix
        c = absorbance.shape[0]
        absorbance = absorbance.reshape((c, -1))

        # remove transparent pixels
        absorbance = absorbance[np.all(absorbance > self.beta, axis=1)]
        if len(absorbance) == 0:
            raise ValueError("All pixels of the input image are below the absorbance threshold.")

        return absorbance

    def _stain_decomposition(self, absorbance: np.ndarray) -> np.ndarray:
        """Calculate the matrix of stain coefficient from the image.

        Args:
            absorbance: absorbance matrix to perform stain extraction on

        Return:
            stain_coeff: stain attenuation coefficient matrix derive from the
                image, where first column is H, second column is E, and
                rows are RGB values
        """

        # compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(absorbance).astype(np.float32, copy=False))

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        projection = np.dot(eigvecs[:, -2:].T, absorbance)

        # find the vectors that span the whole data (min and max angles)
        phi = np.arctan2(projection[1], projection[0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        # project back to absorbance space
        v_min = eigvecs[:, -2:].dot(np.array([(np.cos(min_phi), np.sin(min_phi))], dtype=np.float32).T)
        v_max = eigvecs[:, -2:].dot(np.array([(np.cos(max_phi), np.sin(max_phi))], dtype=np.float32).T)

        # make the vector corresponding to hematoxylin first and eosin second (based on R channel)
        if v_min[0] > v_max[0]:
            stain_coeff = np.array((v_min[:, 0], v_max[:, 0]), dtype=np.float32).T
        else:
            stain_coeff = np.array((v_max[:, 0], v_min[:, 0]), dtype=np.float32).T

        return stain_coeff

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Perform stain extraction.

        Args:
            image: RGB image to extract stain from

        Return:
            ref_stain_coeff: H&E absorbance matrix for the image (first column is H, second column is E, rows are RGB values)
        """
        # check image type and values
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be of type cupy.ndarray.")
        if image.min() < 0:
            raise ValueError("Image should not have negative values.")

        absorbance = self.calculate_flat_absorbance(image)
        ref_stain_coeff = self._stain_decomposition(absorbance)
        return ref_stain_coeff


class StainNormalizer(Transform):
    """Normalize images to a reference stain color matrix.

    First, it extracts the stain coefficient matrix from the image using the provided stain extractor.
    Then, it calculates the stain concentrations based on Beer-Lamber Law.
    Next, it reconstructs the image using the provided reference stain matrix (stain-normalized image).

    Args:
        source_intensity: transmitted light intensity.
            Defaults to 240.
        alpha: percentiles to ignore for outliers, so to calculate min and max,
            if only consider (alpha, 100-alpha) percentiles. Defaults to 1.
        ref_stain_coeff: reference stain attenuation coefficient matrix.
            Defaults to ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)).
        ref_max_conc: reference maximum stain concentrations for
            Hematoxylin & Eosin (H&E). Defaults to (1.9705, 1.0308).

    """

    def __init__(
        self,
        source_intensity: float = 240,
        alpha: float = 1,
        ref_stain_coeff: Union[tuple, np.ndarray] = ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)),
        ref_max_conc: Union[tuple, np.ndarray] = (1.9705, 1.0308),
        stain_extractor=None,
    ) -> None:
        self.source_intensity = source_intensity
        self.alpha = alpha
        self.ref_stain_coeff = np.array(ref_stain_coeff)
        self.ref_max_conc = np.array(ref_max_conc)
        if stain_extractor is None:
            self.stain_extractor = HEStainExtractor()
        else:
            self.stain_extractor = stain_extractor

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Perform stain normalization.

        Args:
            image: uint8 RGB image to be stain normalized, pixel values between 0 and 255

        Return:
            image_norm: stain normalized image/patch
        """
        # check image type and values
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be of type cupy.ndarray.")
        if image.min() < 0:
            raise ValueError("Image should not have negative values.")

        if self.source_intensity < 0:
            raise ValueError("Source transmitted light intensity must be a positive value.")

        # derive stain coefficient matrix from the image
        stain_coeff = self.stain_extractor(image)

        # calculate absorbance
        image = image.astype(np.float32, copy=False) + 1.0
        absorbance = -np.log(image.clip(max=self.source_intensity) / self.source_intensity)

        # reshape to form a CxN matrix
        c, h, w = absorbance.shape
        absorbance = absorbance.reshape((c, -1))

        # calculate concentrations of the each stain, based on Beer-Lambert Law
        conc_raw = np.linalg.lstsq(stain_coeff, absorbance, rcond=None)[0]

        # normalize stain concentrations
        max_conc = np.percentile(conc_raw, 100 - self.alpha, axis=1)
        normalization_factors = self.ref_max_conc / max_conc
        conc_norm = conc_raw * normalization_factors[:, np.newaxis]

        # reconstruct the image based on the reference stain matrix
        image_norm: np.ndarray = np.multiply(
            self.source_intensity, np.exp(-self.ref_stain_coeff.dot(conc_norm)), dtype=np.float32
        )
        image_norm = np.reshape(image_norm, (c, h, w)).astype(np.uint8)
        return image_norm
