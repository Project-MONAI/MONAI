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


def write_nifti(data, affine, file_name, revert_canonical, original_affine=None, dtype="float32"):
    """Write numpy data into nifti files to disk.

    Args:
        data (numpy.ndarray): input data to write to file.
        affine (numpy.ndarray): affine information for the data.
        file_name (string): expected file name that saved on disk.
        revert_canonical (bool): whether to revert canonical.
        original_affine (numpy.ndarray, optional): affine matrix before canonical transformation. Required if revert_canonical.
        dtype (np.dtype, optional): convert the image to save to this data type.

    """
    assert isinstance(data, np.ndarray), 'input data must be numpy array.'
    if affine is None:
        affine = np.eye(4)

    if revert_canonical:
        if not original_affine:
            raise ValueError("original_affine must be passed if revert_canonical is True.")

        ornt = nib.orientations.io_orientation(original_affine)
        # apply same orientation twice to cancel the behavior.
        reverted_results = nib.orientations.apply_orientation(np.squeeze(data), ornt)
        results_img = nib.Nifti1Image(reverted_results.astype(self._dtype), original_affine)

    else:
        results_img = nib.Nifti1Image(np.squeeze(data).astype(dtype), np.squeeze(affine))

    nib.save(results_img, file_name)
