# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import nibabel as nib
from .multi_format_transformer import MultiFormatTransformer


class NiftiWriter(MultiFormatTransformer):
    """Write nifti files to disk.

    Args:
        use_identity (bool): If true, affine matrix of data is ignored. (Default: False)
        compressed (bool): Should save in compressed format. (Default: True)
    """

    def __init__(self, dtype="float32", use_identity=False, compressed=True):
        MultiFormatTransformer.__init__(self)
        self._dtype = dtype
        self._use_identity = use_identity
        self._compressed = compressed

    def _handle_chw(self, img):
        # convert to channels-last
        return np.transpose(img, (1, 2, 0))

    def _handle_chwd(self, img):
        # convert to channels-last
        return np.transpose(img, (1, 2, 3, 0))

    def _write_file(self, data, affine, file_name, revert_canonical):
        if affine is None:
            affine = np.eye(4)

        if revert_canonical:
            codes = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(np.linalg.inv(affine)))
            reverted_results = nib.orientations.apply_orientation(np.squeeze(data), codes)
            results_img = nib.Nifti1Image(reverted_results.astype(self._dtype), affine)
        else:
            results_img = nib.Nifti1Image(np.squeeze(data).astype(self._dtype), np.squeeze(affine))

        nib.save(results_img, file_name)

    def write(self, img, affine, revert_canonical: bool, file_basename: str):
        """Write Nifti file from given data.

        Args:
            img: image data.
            affine: the affine matrix
            revert_canonical: whether to revert canonical when writing the file
            file_basename (str): path for written nifti file.

        Returns:
        """
        assert isinstance(file_basename, str), 'file_basename must be str'
        assert file_basename, 'file_basename must not be empty'

        file_name = file_basename
        if self._compressed:
            file_name = file_basename + ".nii.gz"

        # create and save the nifti image
        # check for existing affine matrix from LoadNifti
        if self._use_identity:
            affine = None

        if affine:
            assert affine.shape == (4, 4), \
                'Affine must shape (4, 4) but is shape {}'.format(affine.shape)

        img = self.transform(img)
        self._write_file(img, affine, file_name, revert_canonical)
