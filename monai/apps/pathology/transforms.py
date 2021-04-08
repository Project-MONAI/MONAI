# modified from sources:
# - Original implementation from Macenko paper in Matlab: https://github.com/mitkovetta/staining-normalization
# - Implementation in Python: https://github.com/schaugf/HEnorm_python
# - Link to Macenko et al., 2009 paper: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
from monai.transforms.transform import Transform
from monai.utils import exact_version, optional_import

cp, _ = optional_import("cupy", "8.6.0", exact_version)


class ExtractStainsMacenko(Transform):
    """Class to extract a target stain from an image, using the Macenko method for stain deconvolution.

    Args:
        tli: (optional) transmitted light intensity
        alpha: (optional) tolerance for the pseudo-min and pseudo-max
        beta: (optional) Optical Density (OD) threshold for transparent pixels
        max_cref: (optional) reference maximum stain concentrations for Hematoxylin & Eosin (H&E)
    """

    def __init__(self, tli: float = 240, alpha: float = 1, beta: float = 0.15, max_cref: cp.ndarray = None) -> None:
        self.tli = tli
        self.alpha = alpha
        self.beta = beta

        self.max_cref = max_cref
        if self.max_cref is None:
            self.max_cref = cp.array([1.9705, 1.0308])

    def _deconvolution_extract_stain(self, img: cp.ndarray) -> cp.ndarray:
        """Perform Stain Deconvolution using the Macenko Method, and return stain matrix for the image.

        Args:
            img: RGB image to perform stain deconvolution of

        Return:
            he: H&E OD matrix for the image (first column is H, second column is E, rows are RGB values)
        """
        # reshape image
        img = img.reshape((-1, 3))

        # calculate optical density
        od = -cp.log((img.astype(cp.float) + 1) / self.tli)

        # remove transparent pixels
        od_hat = od[~cp.any(od < self.beta, axis=1)]

        # compute eigenvectors
        _, eigvecs = cp.linalg.eigh(cp.cov(od_hat.T))

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        t_hat = od_hat.dot(eigvecs[:, 1:3])

        # find the min and max vectors and project back to OD space
        phi = cp.arctan2(t_hat[:, 1], t_hat[:, 0])
        min_phi = cp.percentile(phi, self.alpha)
        max_phi = cp.percentile(phi, 100 - self.alpha)
        v_min = eigvecs[:, 1:3].dot(cp.array([(cp.cos(min_phi), cp.sin(min_phi))]).T)
        v_max = eigvecs[:, 1:3].dot(cp.array([(cp.cos(max_phi), cp.sin(max_phi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        if v_min[0] > v_max[0]:
            he = cp.array((v_min[:, 0], v_max[:, 0])).T
        else:
            he = cp.array((v_max[:, 0], v_min[:, 0])).T

        return he

    def __call__(self, image: cp.ndarray) -> cp.ndarray:
        """Perform stain extraction.

        Args:
            image: RGB image to extract stain from

        return:
            target_he: H&E OD matrix for the image (first column is H, second column is E, rows are RGB values)
        """
        target_he = self._deconvolution_extract_stain(image)
        return target_he


class NormalizeStainsMacenko(Transform):
    """Class to normalize patches/images to a reference or target image stain, using the Macenko method.

    Performs stain deconvolution of the source image to obtain the stain concentration matrix
    for the image. Then, performs the inverse Beer-Lambert transform to recreate the
    patch using the target H&E stain matrix provided. If no target stain provided, a default
    reference stain is used. Similarly, if no maximum stain concentrations are provided, a
    reference maximum stain concentrations matrix is used.

    Args:
        tli: (optional) transmitted light intensity
        alpha: (optional) tolerance for the pseudo-min and pseudo-max
        beta: (optional) Optical Density (OD) threshold for transparent pixels
        target_he: (optional) target stain matrix
        max_cref: (optional) reference maximum stain concentrations for Hematoxylin & Eosin (H&E)
    """

    def __init__(
        self,
        tli: float = 240,
        alpha: float = 1,
        beta: float = 0.15,
        target_he: cp.ndarray = None,
        max_cref: cp.ndarray = None,
    ) -> None:
        self.tli = tli
        self.alpha = alpha
        self.beta = beta

        self.target_he = target_he
        if self.target_he is None:
            self.target_he = cp.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])

        self.max_cref = max_cref
        if self.max_cref is None:
            self.max_cref = cp.array([1.9705, 1.0308])

    def _deconvolution_extract_conc(self, img: cp.ndarray) -> cp.ndarray:
        """Perform Stain Deconvolution using the Macenko Method, and return stain concentration.

        Args:
            img: RGB image to perform stain deconvolution of

        Return:
            conc_norm: stain concentration matrix for the input image
        """
        # reshape image
        img = img.reshape((-1, 3))

        # calculate optical density
        od = -cp.log((img.astype(cp.float) + 1) / self.tli)

        # remove transparent pixels
        od_hat = od[~cp.any(od < self.beta, axis=1)]

        # compute eigenvectors
        _, eigvecs = cp.linalg.eigh(cp.cov(od_hat.T))

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        t_hat = od_hat.dot(eigvecs[:, 1:3])

        # find the min and max vectors and project back to OD space
        phi = cp.arctan2(t_hat[:, 1], t_hat[:, 0])
        min_phi = cp.percentile(phi, self.alpha)
        max_phi = cp.percentile(phi, 100 - self.alpha)
        v_min = eigvecs[:, 1:3].dot(cp.array([(cp.cos(min_phi), cp.sin(min_phi))]).T)
        v_max = eigvecs[:, 1:3].dot(cp.array([(cp.cos(max_phi), cp.sin(max_phi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        if v_min[0] > v_max[0]:
            he = cp.array((v_min[:, 0], v_max[:, 0])).T
        else:
            he = cp.array((v_max[:, 0], v_min[:, 0])).T

        # rows correspond to channels (RGB), columns to OD values
        y = cp.reshape(od, (-1, 3)).T

        # determine concentrations of the individual stains
        conc = cp.linalg.lstsq(he, y, rcond=None)[0]

        # normalize stain concentrations
        max_conc = cp.array([cp.percentile(conc[0, :], 99), cp.percentile(conc[1, :], 99)])
        tmp = cp.divide(max_conc, self.max_cref)
        conc_norm = cp.divide(conc, tmp[:, cp.newaxis])
        return conc_norm

    def __call__(self, image: cp.ndarray) -> cp.ndarray:
        """Perform stain normalization.

        Args:
            image: RGB image/patch to stain normalize

        Return:
            image_norm: stain normalized image/patch
        """
        h, w, _ = image.shape
        image_c = self._deconvolution_extract_conc(image)

        image_norm = cp.multiply(self.tli, cp.exp(-self.target_he.dot(image_c)))
        image_norm[image_norm > 255] = 254
        image_norm = cp.reshape(image_norm.T, (h, w, 3)).astype(cp.uint8)
        return image_norm
