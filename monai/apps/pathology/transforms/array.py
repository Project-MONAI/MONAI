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


from typing import TYPE_CHECKING

from monai.transforms.transform import Transform
from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    import cupy as cp
    from cupy import ndarray as cp_ndarray
else:
    cp, _ = optional_import("cupy", "8.6.0", exact_version)
    cp_ndarray, _ = optional_import("cupy", "8.6.0", exact_version, name="ndarray")


class ExtractHEStains(Transform):
    """Class to extract a target stain from an image, using the Macenko method for stain deconvolution.

    Args:
        tli: transmitted light intensity. Defaults to 240.
        alpha: tolerance in percentile for the pseudo-min (alpha percentile)
            and pseudo-max (100 - alpha percentile). Defaults to 1.
        beta: absorbance threshold for transparent pixels. Defaults to 0.15
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to None.

    Note:
        For more information refer to:
        - the original paper: Macenko et al., 2009 http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
        - the previous implementations:
            - MATLAB: https://github.com/mitkovetta/staining-normalization
            - Python: https://github.com/schaugf/HEnorm_python
    """

    def __init__(self, tli: float = 240, alpha: float = 1, beta: float = 0.15, max_cref: cp_ndarray = None) -> None:
        self.tli = tli
        self.alpha = alpha
        self.beta = beta

        self.max_cref = max_cref
        if self.max_cref is None:
            self.max_cref = cp.array([1.9705, 1.0308])

    def _deconvolution_extract_stain(self, img: cp_ndarray) -> cp_ndarray:
        """Perform Stain Deconvolution using the Macenko Method, and return stain matrix for the image.

        Args:
            img: uint8 RGB image to perform stain deconvolution of

        Return:
            he: H&E absorbance matrix for the image (first column is H, second column is E, rows are RGB values)
        """
        # reshape image
        img = img.reshape((-1, 3))

        # calculate absorbance
        absorbance = -cp.log(cp.clip(img.astype(cp.float32) + 1, a_max=self.tli) / self.tli)

        # remove transparent pixels
        absorbance_hat = absorbance[cp.all(absorbance > self.beta, axis=1)]
        if len(absorbance_hat) == 0:
            raise ValueError("All pixels of the input image are below the absorbance threshold.")

        # compute eigenvectors
        _, eigvecs = cp.linalg.eigh(cp.cov(absorbance_hat.T).astype(cp.float32))

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        t_hat = absorbance_hat.dot(eigvecs[:, 1:3])

        # find the min and max vectors and project back to absorbance space
        phi = cp.arctan2(t_hat[:, 1], t_hat[:, 0])
        min_phi = cp.percentile(phi, self.alpha)
        max_phi = cp.percentile(phi, 100 - self.alpha)
        v_min = eigvecs[:, 1:3].dot(cp.array([(cp.cos(min_phi), cp.sin(min_phi))], dtype=cp.float32).T)
        v_max = eigvecs[:, 1:3].dot(cp.array([(cp.cos(max_phi), cp.sin(max_phi))], dtype=cp.float32).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        if v_min[0] > v_max[0]:
            he = cp.array((v_min[:, 0], v_max[:, 0]), dtype=cp.float32).T
        else:
            he = cp.array((v_max[:, 0], v_min[:, 0]), dtype=cp.float32).T

        return he

    def __call__(self, image: cp_ndarray) -> cp_ndarray:
        """Perform stain extraction.

        Args:
            image: uint8 RGB image to extract stain from

        return:
            target_he: H&E absorbance matrix for the image (first column is H, second column is E, rows are RGB values)
        """
        if not isinstance(image, cp_ndarray):
            raise TypeError("Image must be of type cupy.ndarray.")

        target_he = self._deconvolution_extract_stain(image)
        return target_he


class NormalizeStainsMacenko(Transform):
    """Class to normalize patches/images to a reference or target image stain, using the Macenko method.

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
        target_he: target stain matrix. Defaults to None.
        max_cref: reference maximum stain concentrations for Hematoxylin & Eosin (H&E).
            Defaults to None.

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
        target_he: cp_ndarray = None,
        max_cref: cp_ndarray = None,
    ) -> None:
        self.tli = tli

        self.target_he = target_he
        if self.target_he is None:
            self.target_he = cp.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])

        self.max_cref = max_cref
        if self.max_cref is None:
            self.max_cref = cp.array([1.9705, 1.0308])

        self.stain_extractor = ExtractHEStains(tli=self.tli, alpha=alpha, beta=beta, max_cref=self.max_cref)

    def __call__(self, image: cp_ndarray) -> cp_ndarray:
        """Perform stain normalization.

        Args:
            image: uint8 RGB image/patch to stain normalize

        Return:
            image_norm: stain normalized image/patch
        """
        if not isinstance(image, cp_ndarray):
            raise TypeError("Image must be of type cupy.ndarray.")

        # extract stain of the image
        he = self.stain_extractor(image)

        h, w, _ = image.shape

        # reshape image and calculate absorbance
        image = image.reshape((-1, 3))
        absorbance = -cp.log(cp.clip(image.astype(cp.float32) + 1, a_max=self.tli) / self.tli)

        # rows correspond to channels (RGB), columns to absorbance values
        y = cp.reshape(absorbance, (-1, 3)).T

        # determine concentrations of the individual stains
        conc = cp.linalg.lstsq(he, y, rcond=None)[0]

        # normalize stain concentrations
        max_conc = cp.array([cp.percentile(conc[0, :], 99), cp.percentile(conc[1, :], 99)], dtype=cp.float32)
        tmp = cp.divide(max_conc, self.max_cref, dtype=cp.float32)
        image_c = cp.divide(conc, tmp[:, cp.newaxis], dtype=cp.float32)

        image_norm = cp.multiply(self.tli, cp.exp(-self.target_he.dot(image_c)), dtype=cp.float32)
        image_norm[image_norm > 255] = 254
        image_norm = cp.reshape(image_norm.T, (h, w, 3)).astype(cp.uint8)
        return image_norm
