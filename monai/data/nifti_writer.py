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
import torch

from monai.data.utils import compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform


def write_nifti(
    data,
    file_name: str,
    affine=None,
    target_affine=None,
    resample: bool = True,
    output_shape=None,
    interp_order: str = "bilinear",
    mode: str = "border",
    dtype=None,
):
    """
    Write numpy data into NIfTI files to disk.  This function converts data
    into the coordinate system defined by `target_affine` when `target_affine`
    is specified.

    If the coordinate transform between `affine` and `target_affine` could be
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

    When `affine` and `target_affine` are None, the data will be saved with an
    identity matrix as the image affine.

    This function assumes the NIfTI dimension notations.
    Spatially it supports up to three dimensions, that is, H, HW, HWD for
    1D, 2D, 3D respectively.
    When saving multiple time steps or multiple channels `data`, time and/or
    modality axes should be appended after the first three dimensions.  For
    example, shape of 2D eight-class segmentation probabilities to be saved
    could be `(64, 64, 1, 8)`. Also, data in shape (64, 64, 8), (64, 64, 8, 1)
    will be considered as a single-channel 3D image.

    Args:
        data (numpy.ndarray): input data to write to file.
        file_name: expected file name that saved on disk.
        affine (numpy.ndarray): the current affine of `data`. Defaults to `np.eye(4)`
        target_affine (numpy.ndarray, optional): before saving
            the (`data`, `affine`) as a Nifti1Image,
            transform the data into the coordinates defined by `target_affine`.
        resample: whether to run resampling when the target affine
            could not be achieved by swapping/flipping data axes.
        output_shape (None or tuple of ints): output image shape.
            This option is used when resample = True.
        interp_order (`nearest|bilinear`): the interpolation mode, default is "bilinear".
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            This option is used when `resample = True`.
        mode (`zeros|border|reflection`):
            The mode parameter determines how the input array is extended beyond its boundaries.
            Defaults to "border". This option is used when `resample = True`.
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

    if np.allclose(affine, target_affine, atol=1e-3):
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
    if np.allclose(_affine, target_affine, atol=1e-3) or not resample:
        results_img = nib.Nifti1Image(data.astype(dtype), to_affine_nd(3, target_affine))
        nib.save(results_img, file_name)
        return

    # need resampling
    affine_xform = AffineTransform(
        normalized=False, mode=interp_order, padding_mode=mode, align_corners=True, reverse_indexing=True
    )
    transform = np.linalg.inv(_affine) @ target_affine
    if output_shape is None:
        output_shape, _ = compute_shape_offset(data.shape, _affine, target_affine)
    if data.ndim > 3:  # multi channel, resampling each channel
        while len(output_shape) < 3:
            output_shape = list(output_shape) + [1]
        spatial_shape, channel_shape = data.shape[:3], data.shape[3:]
        data_ = data.reshape(list(spatial_shape) + [-1])
        data_ = np.moveaxis(data_, -1, 0)  # channel first for pytorch
        data_ = affine_xform(
            torch.from_numpy((data_.astype(np.float64))[None]),
            torch.from_numpy(transform.astype(np.float64)),
            spatial_size=output_shape[:3],
        )
        data_ = data_.squeeze(0).detach().cpu().numpy()
        data_ = np.moveaxis(data_, 0, -1)  # channel last for nifti
        data_ = data_.reshape(list(data_.shape[:3]) + list(channel_shape))
    else:  # single channel image, need to expand to have batch and channel
        while len(output_shape) < len(data.shape):
            output_shape = list(output_shape) + [1]
        data_ = affine_xform(
            torch.from_numpy((data.astype(np.float64))[None, None]),
            torch.from_numpy(transform.astype(np.float64)),
            spatial_size=output_shape[: len(data.shape)],
        )
        data_ = data_.squeeze(0).squeeze(0).detach().cpu().numpy()
    dtype = dtype or data.dtype
    results_img = nib.Nifti1Image(data_.astype(dtype), to_affine_nd(3, target_affine))
    nib.save(results_img, file_name)
    return
