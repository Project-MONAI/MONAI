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

from typing import Optional, Sequence, Union

import numpy as np
import torch

from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.utils import GridSampleMode, GridSamplePadMode, optional_import
from monai.utils.type_conversion import convert_data_type

nib, _ = optional_import("nibabel")


def write_nifti(
    data: NdarrayOrTensor,
    file_name: str,
    affine: Optional[NdarrayOrTensor] = None,
    target_affine: Optional[np.ndarray] = None,
    resample: bool = True,
    output_spatial_shape: Union[Sequence[int], np.ndarray, None] = None,
    mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
    padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
    align_corners: bool = False,
    dtype: DtypeLike = np.float64,
    output_dtype: DtypeLike = np.float32,
) -> None:
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
    13.333x13.333 pixels. In this case `output_spatial_shape` could be specified so
    that this function writes image data to a designated shape.

    The saved `affine` matrix follows:
    - If `affine` equals to `target_affine`, save the data with `target_affine`.
    - If `resample=False`, transform `affine` to `new_affine` based on the orientation
    of `target_affine` and save the data with `new_affine`.
    - If `resample=True`, save the data with `target_affine`, if explicitly specify
    the `output_spatial_shape`, the shape of saved data is not computed by `target_affine`.
    - If `target_affine` is None, set `target_affine=affine` and save.
    - If `affine` and `target_affine` are None, the data will be saved with an identity
    matrix as the image affine.

    This function assumes the NIfTI dimension notations.
    Spatially it supports up to three dimensions, that is, H, HW, HWD for
    1D, 2D, 3D respectively.
    When saving multiple time steps or multiple channels `data`, time and/or
    modality axes should be appended after the first three dimensions.  For
    example, shape of 2D eight-class segmentation probabilities to be saved
    could be `(64, 64, 1, 8)`. Also, data in shape (64, 64, 8), (64, 64, 8, 1)
    will be considered as a single-channel 3D image.

    Args:
        data: input data to write to file.
        file_name: expected file name that saved on disk.
        affine: the current affine of `data`. Defaults to `np.eye(4)`
        target_affine: before saving
            the (`data`, `affine`) as a Nifti1Image,
            transform the data into the coordinates defined by `target_affine`.
        resample: whether to run resampling when the target affine
            could not be achieved by swapping/flipping data axes.
        output_spatial_shape: spatial shape of the output image.
            This option is used when resample = True.
        mode: {``"bilinear"``, ``"nearest"``}
            This option is used when ``resample = True``.
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            This option is used when ``resample = True``.
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
            If None, use the data type of input data.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
    """
    if isinstance(data, torch.Tensor):
        data, *_ = convert_data_type(data, np.ndarray)
    if isinstance(affine, torch.Tensor):
        affine, *_ = convert_data_type(affine, np.ndarray)
    if not isinstance(data, np.ndarray):
        raise AssertionError("input data must be numpy array or torch tensor.")
    dtype = dtype or data.dtype
    sr = min(data.ndim, 3)
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
    affine = to_affine_nd(sr, affine)  # type: ignore

    if target_affine is None:
        target_affine = affine
    target_affine = to_affine_nd(sr, target_affine)

    if np.allclose(affine, target_affine, atol=1e-3):
        # no affine changes, save (data, affine)
        results_img = nib.Nifti1Image(data.astype(output_dtype, copy=False), to_affine_nd(3, target_affine))
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
        results_img = nib.Nifti1Image(data.astype(output_dtype, copy=False), to_affine_nd(3, _affine))  # type: ignore
        nib.save(results_img, file_name)
        return

    # need resampling
    affine_xform = AffineTransform(
        normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
    )
    transform = np.linalg.inv(_affine) @ target_affine
    if output_spatial_shape is None:
        output_spatial_shape, _ = compute_shape_offset(data.shape, _affine, target_affine)
    output_spatial_shape_ = list(output_spatial_shape) if output_spatial_shape is not None else []
    if data.ndim > 3:  # multi channel, resampling each channel
        while len(output_spatial_shape_) < 3:
            output_spatial_shape_ = output_spatial_shape_ + [1]
        spatial_shape, channel_shape = data.shape[:3], data.shape[3:]
        data_np: np.ndarray = data.reshape(list(spatial_shape) + [-1])  # type: ignore
        data_np = np.moveaxis(data_np, -1, 0)  # channel first for pytorch
        data_torch = affine_xform(
            torch.as_tensor(np.ascontiguousarray(data_np, dtype=dtype)).unsqueeze(0),
            torch.as_tensor(np.ascontiguousarray(transform, dtype=dtype)),
            spatial_size=output_spatial_shape_[:3],
        )
        data_np = data_torch.squeeze(0).detach().cpu().numpy()
        data_np = np.moveaxis(data_np, 0, -1)  # channel last for nifti
        data_np = data_np.reshape(list(data_np.shape[:3]) + list(channel_shape))
    else:  # single channel image, need to expand to have batch and channel
        while len(output_spatial_shape_) < len(data.shape):
            output_spatial_shape_ = output_spatial_shape_ + [1]
        data_torch = affine_xform(
            torch.as_tensor(np.ascontiguousarray(data, dtype=dtype)[None, None]),
            torch.as_tensor(np.ascontiguousarray(transform, dtype=dtype)),
            spatial_size=output_spatial_shape_[: len(data.shape)],
        )
        data_np = data_torch.squeeze(0).squeeze(0).detach().cpu().numpy()

    results_img = nib.Nifti1Image(data_np.astype(output_dtype, copy=False), to_affine_nd(3, target_affine))
    nib.save(results_img, file_name)
    return
