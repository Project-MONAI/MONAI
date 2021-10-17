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

from typing import TYPE_CHECKING, Union

import numpy as np

from monai.transforms.transform import Transform
from monai.utils import optional_import

cp, has_cp = optional_import("cupy")


class ExtractHEStains(Transform):
    """Class to extract a target stain from an image, using stain deconvolution (see Note).

    Args:
        tli: transmitted light intensity. Defaults to 240.
        alpha: tolerance in percentile for the pseudo-min (alpha percentile)
            and pseudo-max (100 - alpha percentile). Defaults to 1.
        beta: absorbance threshold for transparent pixels. Defaults to 0.15
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to (1.9705, 1.0308).

    Note:
        For more information refer to:
        - the original paper: Macenko et al., 2009 http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
        - the previous implementations:

          - MATLAB: https://github.com/mitkovetta/staining-normalization
          - Python: https://github.com/schaugf/HEnorm_python
    """

    def __init__(
        self,
        tli: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        max_cref: tuple = (1.9705, 1.0308),
        backend: str = "numpy",
    ) -> None:
        # set backend to numpy or cupy
        self.backend, has_it = optional_import(backend)
        if not has_it:
            raise ValueError(f"The requested backend [{backend}] is not installed!")
        self.tli = tli
        self.alpha = alpha
        self.beta = beta
        self.max_cref = self.backend.array(max_cref)

    def _deconvolution_extract_stain(self, image):
        """Perform Stain Deconvolution and return stain matrix for the image.

        Args:
            img: numpy.ndarray or cupy.ndarray
                uint8 RGB image to perform stain deconvolution on

        Return:
            he: numpy.ndarray or cupy.ndarray
                H&E absorbance matrix for the image (first column is H, second column is E, rows are RGB values)
        """
        # check image type and values
        if not isinstance(image, self.backend.ndarray):
            raise TypeError("Image must be of type numpy.ndarray.")
        if image.min() < 0:
            raise ValueError("Image should not have negative values.")
        if image.max() > 255:
            raise ValueError("Image should not have values greater than 255.")

        # reshape image and calculate absorbance
        image = image.reshape((-1, 3))
        image = image.astype(self.backend.float32) + 1.0
        absorbance = -self.backend.log(self.backend.clip(image, a_min=None, a_max=self.tli) / self.tli)

        # remove transparent pixels
        absorbance_hat = absorbance[self.backend.all(absorbance > self.beta, axis=1)]
        if len(absorbance_hat) == 0:
            raise ValueError("All pixels of the input image are below the absorbance threshold.")

        # compute eigenvectors
        _, eigvecs = self.backend.linalg.eigh(self.backend.cov(absorbance_hat.T).astype(self.backend.float32))

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        t_hat = absorbance_hat.dot(eigvecs[:, 1:3])

        # find the min and max vectors and project back to absorbance space
        phi = self.backend.arctan2(t_hat[:, 1], t_hat[:, 0])
        min_phi = self.backend.percentile(phi, self.alpha)
        max_phi = self.backend.percentile(phi, 100 - self.alpha)
        v_min = eigvecs[:, 1:3].dot(
            self.backend.array([(self.backend.cos(min_phi), self.backend.sin(min_phi))], dtype=self.backend.float32).T
        )
        v_max = eigvecs[:, 1:3].dot(
            self.backend.array([(self.backend.cos(max_phi), self.backend.sin(max_phi))], dtype=self.backend.float32).T
        )

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        if v_min[0] > v_max[0]:
            he = self.backend.array((v_min[:, 0], v_max[:, 0]), dtype=self.backend.float32).T
        else:
            he = self.backend.array((v_max[:, 0], v_min[:, 0]), dtype=self.backend.float32).T

        return he

    def __call__(self, image):
        """Perform stain extraction.

        Args:
            image: numpy.ndarray or cupy.ndarray
                uint8 RGB image to extract stain from numpy.ndarray or cupy.ndarray

        return:
            target_he: numpy.ndarray or cupy.ndarray
                H&E absorbance matrix for the image (first column is H, second column is E, rows are RGB values)

        """
        if not isinstance(image, self.backend.ndarray):
            raise TypeError("Image must be of type numpy.ndarray.")

        target_he = self._deconvolution_extract_stain(image)
        return target_he


class NormalizeHEStains(Transform):
    """Class to normalize patches/images to a reference or target image stain (see Note).

    Performs stain deconvolution of the source image using the ExtractHEStains
    class, to obtain the stain matrix and calculate the stain concentration matrix
    for the image. Then, performs the inverse Beer-Lambert transform to recreate the
    patch using the target H&E stain matrix provided. If no target stain provided, a default
    reference stain is used. Similarly, if no maximum stain concentrations are provided, a
    reference maximum stain concentrations matrix is used.

    Args:
        tli: transmitted light intensity. Defaults to 240.
        alpha: tolerance in percentile for the pseudo-min (alpha percentile) and
            pseudo-max (100 - alpha percentile). Defaults to 1.
        beta: absorbance threshold for transparent pixels. Defaults to 0.15.
        target_he: target stain matrix. Defaults to ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)).
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to [1.9705, 1.0308].

    Note:
        For more information refer to:
        - the original paper: Macenko et al., 2009 http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
        - the previous implementations:

            - MATLAB: https://github.com/mitkovetta/staining-normalization
            - Python: https://github.com/schaugf/HEnorm_python
    """

    def __init__(
        self,
        tli: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        target_he: Union[tuple, np.ndarray] = ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581)),
        max_cref: tuple = (1.9705, 1.0308),
        backend: str = "numpy",
    ) -> None:
        # set backend to numpy or cupy
        self.backend, has_it = optional_import(backend)
        if not has_it:
            raise ValueError(f"The requested backend [{backend}] is not installed!")
        self.tli = tli
        self.target_he = self.backend.array(target_he)
        self.max_cref = self.backend.array(max_cref)
        self.stain_extractor = ExtractHEStains(
            tli=self.tli,
            alpha=alpha,
            beta=beta,
            max_cref=self.max_cref,
            backend=backend,
        )

    def __call__(self, image):
        """Perform stain normalization.

        Args:
            image: numpy.ndarray or cupy.ndarray
                uint8 RGB image/patch to be stain normalized, pixel values between 0 and 255

        Return:
            image_norm: numpy.ndarray or cupy.ndarray
                stain normalized image/patch
        """
        # check image type and values
        if not isinstance(image, self.backend.ndarray):
            raise TypeError("Image must be of type numpy.ndarray.")
        if image.min() < 0:
            raise ValueError("Image should not have negative values.")
        if image.max() > 255:
            raise ValueError("Image should not have values greater than 255.")

        # extract stain of the image
        he = self.stain_extractor(image)

        # reshape image and calculate absorbance
        h, w, _ = image.shape
        image = image.reshape((-1, 3))
        image = image.astype(self.backend.float32) + 1.0
        absorbance = -self.backend.log(self.backend.clip(image, a_min=None, a_max=self.tli) / self.tli)

        # rows correspond to channels (RGB), columns to absorbance values
        y = self.backend.reshape(absorbance, (-1, 3)).T

        # determine concentrations of the individual stains
        conc = self.backend.linalg.lstsq(he, y, rcond=None)[0]

        # normalize stain concentrations
        max_conc = self.backend.array(
            [self.backend.percentile(conc[0, :], 99), self.backend.percentile(conc[1, :], 99)],
            dtype=self.backend.float32,
        )
        tmp = self.backend.divide(max_conc, self.max_cref, dtype=self.backend.float32)
        image_c = self.backend.divide(conc, tmp[:, self.backend.newaxis], dtype=self.backend.float32)

        image_norm: self.backend.ndarray = self.backend.multiply(
            self.tli, self.backend.exp(-self.target_he.dot(image_c)), dtype=self.backend.float32
        )
        image_norm[image_norm > 255] = 254
        image_norm = self.backend.reshape(image_norm.T, (h, w, 3)).astype(self.backend.uint8)
        return image_norm
