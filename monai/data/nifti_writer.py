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


def write_nifti(data, affine, file_name, target_affine=None, dtype=np.float32):
    """Write numpy data into nifti files to disk.

    Args:
        data (numpy.ndarray): input data to write to file.
        affine (numpy.ndarray): affine information for the data.
        file_name (string): expected file name that saved on disk.
        target_affine (numpy.ndarray, optional):
            before saving the (data, affine), transform the data into the orientation defined by `target_affine`.
        dtype (np.dtype, optional): convert the image to save to this data type.
    """
    assert isinstance(data, np.ndarray), 'input data must be numpy array.'
    if affine is None:
        affine = np.eye(4)

    if target_affine is None:
        results_img = nib.Nifti1Image(data.astype(dtype), affine)
    else:
        start_ornt = nib.orientations.io_orientation(affine)
        target_ornt = nib.orientations.io_orientation(target_affine)
        ornt_transform = nib.orientations.ornt_transform(start_ornt, target_ornt)

        reverted_results = nib.orientations.apply_orientation(data, ornt_transform)
        results_img = nib.Nifti1Image(reverted_results.astype(dtype), target_affine)

    nib.save(results_img, file_name)
