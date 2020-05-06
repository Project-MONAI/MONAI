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

import nibabel as nib
import numpy as np
import scipy.ndimage

from monai.data.utils import compute_shape_offset, to_affine_nd


def write_nifti(
    data,
    file_name,
    affine=None,
    target_affine=None,
    resample=True,
    output_shape=None,
    interp_order=3,
    mode="constant",
    cval=0,
    dtype=None,
):
    """
    Write numpy data into NIfTI files to disk.  This function converts data
    into the coordinate system defined by `target_affine` when `target_affine`
    is specified.

    if the coordinate transform between `affine` and `target_affine` could be
    achieved by simply transposing and flipping `data`, no resampling will
    happen.  otherwise this function will resample `data` using the coordinate
    transform computed from `affine` and `target_affine`.  Note that the shape
    of the resampled `data` may subject to some rounding errors. For example,
    resampling a 20x20 pixel image from pixel size (1.5, 1.5)-mm to (3.0,
    3.0)-mm space will return a 10x10-pixel image.  However, resampling a
    20x20-pixel image from pixel size (2.0, 2.0)-mm to (3.0, 3.0)-mma space
    will output a 14x14-pixel image, where the image shape is rounded from
    13.333x13.333 pixels. In this case `output_shape` could be specified so
    that this function writes image data to a designated shape.

    when `affine` and `target_affine` are None, the data will be saved with an
    identity matrix as the image affine.

    This function assumes the NIfTI dimension notations.
    Spatially It supports up to three dimensions, that is, H, HW, HWD for
    1D, 2D, 3D respectively.
    When saving multiple time steps or multiple channels `data`, time and/or
    modality axes should be appended after the first three dimensions.  For
    example, shape of 2D eight-class segmentation probabilities to be saved
    could be `(64, 64, 1, 8)`,

    Args:
        data (numpy.ndarray): input data to write to file.
        file_name (string): expected file name that saved on disk.
        affine (numpy.ndarray): the current affine of `data`. Defaults to `np.eye(4)`
        target_affine (numpy.ndarray, optional): before saving
            the (`data`, `affine`) as a Nifti1Image,
            transform the data into the coordinates defined by `target_affine`.
        resample (bool): whether to run resampling when the target affine
            could not be achieved by swapping/flipping data axes.
        output_shape (None or tuple of ints): output image shape.
            this option is used when resample = True.
        interp_order (int): the order of the spline interpolation, default is 3.
            The order has to be in the range 0 - 5.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
            this option is used when `resample = True`.
        mode (`reflect|constant|nearest|mirror|wrap`):
            The mode parameter determines how the input array is extended beyond its boundaries.
            this option is used when `resample = True`.
        cval (scalar): Value to fill past edges of input if mode is "constant". Default is 0.0.
            this option is used when `resample = True`.
        dtype (np.dtype, optional): convert the image to save to this data type.
    """
    assert isinstance(data, np.ndarray), "input data must be numpy array."
    sr = min(data.ndim, 3)
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
    affine = to_affine_nd(sr, affine)

    if target_affine is None:
        target_affine = affine
    target_affine = to_affine_nd(sr, target_affine)

    if np.allclose(affine, target_affine):
        # no affine changes, save (data, affine)
        results_img = nib.Nifti1Image(data.astype(dtype), to_affine_nd(3, target_affine))
        nib.save(results_img, file_name)
        return

    # resolve orientation
    start_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.io_orientation(target_affine)
    ornt_transform = nib.orientations.ornt_transform(start_ornt, target_ornt)
    data_shape = data.shape
    data = nib.orientations.apply_orientation(data, ornt_transform)
    _affine = affine @ nib.orientations.inv_ornt_aff(ornt_transform, data_shape)
    if np.allclose(_affine, target_affine) or not resample:
        results_img = nib.Nifti1Image(data.astype(dtype), to_affine_nd(3, target_affine))
        nib.save(results_img, file_name)
        return

    # need resampling
    transform = np.linalg.inv(_affine) @ target_affine
    if output_shape is None:
        output_shape, _ = compute_shape_offset(data.shape, _affine, target_affine)
    dtype = dtype or data.dtype
    if data.ndim > 3:  # multi channel, resampling each channel
        spatial_shape, channel_shape = data.shape[:3], data.shape[3:]
        data_ = data.astype(dtype).reshape(list(spatial_shape) + [-1])
        data_chns = []
        for chn in range(data_.shape[-1]):
            data_chns.append(
                scipy.ndimage.affine_transform(
                    data_[..., chn],
                    matrix=transform,
                    output_shape=output_shape[:3],
                    order=interp_order,
                    mode=mode,
                    cval=cval,
                )
            )
        data_chns = np.stack(data_chns, axis=-1)
        data_ = data_chns.reshape(list(data_chns.shape[:3]) + list(channel_shape))
    else:
        data_ = data.astype(dtype)
        data_ = scipy.ndimage.affine_transform(
            data_, matrix=transform, output_shape=output_shape[: data_.ndim], order=interp_order, mode=mode, cval=cval
        )
    results_img = nib.Nifti1Image(data_, to_affine_nd(3, target_affine))
    nib.save(results_img, file_name)
    return
