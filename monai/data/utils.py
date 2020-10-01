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

import math
import os
import warnings
from itertools import product, starmap
from pathlib import PurePath
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from monai.networks.layers.simplelayers import GaussianFilter
from monai.utils import (
    BlendMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    first,
    optional_import,
)

nib, _ = optional_import("nibabel")


def get_random_patch(
    dims: Sequence[int], patch_size: Sequence[int], rand_state: Optional[np.random.RandomState] = None
) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
    close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
    of shape `dims` as returned by `get_valid_patch_size`.

    Args:
        dims: shape of source array
        patch_size: shape of patch size to generate
        rand_state: a random state object to generate random numbers from

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """

    # choose the minimal corner of the patch
    rand_int = np.random.randint if rand_state is None else rand_state.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(dims, patch_size))

    # create the slices for each dimension which define the patch in the source array
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))


def iter_patch_slices(
    dims: Sequence[int], patch_size: Union[Sequence[int], int], start_pos: Sequence[int] = ()
) -> Generator[Tuple[slice, ...], None, None]:
    """
    Yield successive tuples of slices defining patches of size `patch_size` from an array of dimensions `dims`. The
    iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
    patch is chosen in a contiguous grid using a first dimension as least significant ordering.

    Args:
        dims: dimensions of array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension

    Yields:
        Tuples of slice objects defining each patch
    """

    # ensure patchSize and startPos are the right length
    ndim = len(dims)
    patch_size_ = get_valid_patch_size(dims, patch_size)
    start_pos = ensure_tuple_size(start_pos, ndim)

    # collect the ranges to step over each dimension
    ranges = tuple(starmap(range, zip(start_pos, dims, patch_size_)))

    # choose patches by applying product to the ranges
    for position in product(*ranges[::-1]):  # reverse ranges order to iterate in index order
        yield tuple(slice(s, s + p) for s, p in zip(position[::-1], patch_size_))


def dense_patch_slices(
    image_size: Sequence[int],
    patch_size: Sequence[int],
    scan_interval: Sequence[int],
) -> List[Tuple[slice, ...]]:
    """
    Enumerate all slices defining 2D/3D patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval

    Raises:
        ValueError: When ``image_size`` length is not one of [2, 3].

    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    if num_spatial_dims not in (2, 3):
        raise ValueError(f"Unsupported image_size length: {len(image_size)}, available options are [2, 3]")
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = list()
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1)

    slices: List[Tuple[slice, ...]] = []
    if num_spatial_dims == 3:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + patch_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + patch_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + patch_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + patch_size[1])

                for k in range(0, scan_num[2]):
                    start_k = k * scan_interval[2]
                    start_k -= max(start_k + patch_size[2] - image_size[2], 0)
                    slice_k = slice(start_k, start_k + patch_size[2])
                    slices.append((slice_i, slice_j, slice_k))
    else:
        for i in range(scan_num[0]):
            start_i = i * scan_interval[0]
            start_i -= max(start_i + patch_size[0] - image_size[0], 0)
            slice_i = slice(start_i, start_i + patch_size[0])

            for j in range(scan_num[1]):
                start_j = j * scan_interval[1]
                start_j -= max(start_j + patch_size[1] - image_size[1], 0)
                slice_j = slice(start_j, start_j + patch_size[1])
                slices.append((slice_i, slice_j))
    return slices


def iter_patch(
    arr: np.ndarray,
    patch_size: Union[Sequence[int], int] = 0,
    start_pos: Sequence[int] = (),
    copy_back: bool = True,
    mode: Union[NumpyPadMode, str] = NumpyPadMode.WRAP,
    **pad_opts: Dict,
) -> Generator[np.ndarray, None, None]:
    """
    Yield successive patches from `arr` of size `patch_size`. The iteration can start from position `start_pos` in `arr`
    but drawing from a padded array extended by the `patch_size` in each dimension (so these coordinates can be negative
    to start in the padded region). If `copy_back` is True the values from each patch are written back to `arr`.

    Args:
        arr: array to iterate over
        patch_size: size of patches to generate slices for, 0 or None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension
        copy_back: if True data from the yielded patches is copied back to `arr` once the generator completes
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        pad_opts: padding options, see `numpy.pad`

    Yields:
        Patches of array data from `arr` which are views into a padded array which can be modified, if `copy_back` is
        True these changes will be reflected in `arr` once the iteration completes.
    """
    # ensure patchSize and startPos are the right length
    patch_size_ = get_valid_patch_size(arr.shape, patch_size)
    start_pos = ensure_tuple_size(start_pos, arr.ndim)

    # pad image by maximum values needed to ensure patches are taken from inside an image
    arrpad = np.pad(arr, tuple((p, p) for p in patch_size_), NumpyPadMode(mode).value, **pad_opts)

    # choose a start position in the padded image
    start_pos_padded = tuple(s + p for s, p in zip(start_pos, patch_size_))

    # choose a size to iterate over which is smaller than the actual padded image to prevent producing
    # patches which are only in the padded regions
    iter_size = tuple(s + p for s, p in zip(arr.shape, patch_size_))

    for slices in iter_patch_slices(iter_size, patch_size_, start_pos_padded):
        yield arrpad[slices]

    # copy back data from the padded image if required
    if copy_back:
        slices = tuple(slice(p, p + s) for p, s in zip(patch_size_, arr.shape))
        arr[...] = arrpad[slices]


def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)
    patch_size_ = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


def list_data_collate(batch: Sequence):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    elem = batch[0]
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    return default_collate(data)


def worker_init_fn(worker_id: int) -> None:
    """
    Callback function for PyTorch DataLoader `worker_init_fn`.
    It can set different random seed for the transforms in different workers.

    """
    worker_info = torch.utils.data.get_worker_info()
    if hasattr(worker_info.dataset, "transform") and hasattr(worker_info.dataset.transform, "set_random_state"):
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))


def correct_nifti_header_if_necessary(img_nii):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.

    Args:
        img_nii: nifti image object
    """
    dim = img_nii.header["dim"][0]
    if dim >= 5:
        return img_nii  # do nothing for high-dimensional array
    # check that affine matches zooms
    pixdim = np.asarray(img_nii.header.get_zooms())[:dim]
    norm_affine = np.sqrt(np.sum(np.square(img_nii.affine[:dim, :dim]), 0))
    if np.allclose(pixdim, norm_affine):
        return img_nii
    if hasattr(img_nii, "get_sform"):
        return rectify_header_sform_qform(img_nii)
    return img_nii


def rectify_header_sform_qform(img_nii):
    """
    Look at the sform and qform of the nifti object and correct it if any
    incompatibilities with pixel dimensions

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/io/misc_io.py

    Args:
        img_nii: nifti image object
    """
    d = img_nii.header["dim"][0]
    pixdim = np.asarray(img_nii.header.get_zooms())[:d]
    sform, qform = img_nii.get_sform(), img_nii.get_qform()
    norm_sform = np.sqrt(np.sum(np.square(sform[:d, :d]), 0))
    norm_qform = np.sqrt(np.sum(np.square(qform[:d, :d]), 0))
    sform_mismatch = not np.allclose(norm_sform, pixdim)
    qform_mismatch = not np.allclose(norm_qform, pixdim)

    if img_nii.header["sform_code"] != 0:
        if not sform_mismatch:
            return img_nii
        if not qform_mismatch:
            img_nii.set_sform(img_nii.get_qform())
            return img_nii
    if img_nii.header["qform_code"] != 0:
        if not qform_mismatch:
            return img_nii
        if not sform_mismatch:
            img_nii.set_qform(img_nii.get_sform())
            return img_nii

    norm = np.sqrt(np.sum(np.square(img_nii.affine[:d, :d]), 0))
    warnings.warn(f"Modifying image pixdim from {pixdim} to {norm}")

    img_nii.header.set_zooms(norm)
    return img_nii


def zoom_affine(affine: np.ndarray, scale: Sequence[float], diagonal: bool = True) -> np.ndarray:
    """
    To make column norm of `affine` the same as `scale`.  If diagonal is False,
    returns an affine that combines orthogonal rotation and the new scale.
    This is done by first decomposing `affine`, then setting the zoom factors to
    `scale`, and composing a new affine; the shearing factors are removed.  If
    diagonal is True, returns a diagonal matrix, the scaling factors are set
    to the diagonal elements.  This function always return an affine with zero
    translations.

    Args:
        affine (nxn matrix): a square matrix.
        scale: new scaling factor along each dimension.
        diagonal: whether to return a diagonal scaling matrix.
            Defaults to True.

    Raises:
        ValueError: When ``affine`` is not a square matrix.
        ValueError: When ``scale`` contains a nonpositive scalar.

    Returns:
        the updated `n x n` affine.

    """

    affine = np.array(affine, dtype=float, copy=True)
    if len(affine) != len(affine[0]):
        raise ValueError(f"affine must be n x n, got {len(affine)} x {len(affine[0])}.")
    scale_np = np.array(scale, dtype=float, copy=True)
    if np.any(scale_np <= 0):
        raise ValueError("scale must contain only positive numbers.")
    d = len(affine) - 1
    if len(scale_np) < d:  # defaults based on affine
        norm = np.sqrt(np.sum(np.square(affine), 0))[:-1]
        scale_np = np.append(scale_np, norm[len(scale_np) :])
    scale_np = scale_np[:d]
    scale_np[scale_np == 0] = 1.0
    if diagonal:
        return np.diag(np.append(scale_np, [1.0]))
    rzs = affine[:-1, :-1]  # rotation zoom scale
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    s = np.sign(np.diag(zs)) * np.abs(scale_np)
    # construct new affine with rotation and zoom
    new_affine = np.eye(len(affine))
    new_affine[:-1, :-1] = rotation @ np.diag(s)
    return new_affine


def compute_shape_offset(
    spatial_shape: np.ndarray, in_affine: np.ndarray, out_affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given input and output affine, compute appropriate shapes
    in the output space based on the input array's shape.
    This function also returns the offset to put the shape
    in a good position with respect to the world coordinate system.

    Args:
        spatial_shape: input array's shape
        in_affine (matrix): 2D affine matrix
        out_affine (matrix): 2D affine matrix
    """
    shape = np.array(spatial_shape, copy=True, dtype=float)
    sr = len(shape)
    in_affine = to_affine_nd(sr, in_affine)
    out_affine = to_affine_nd(sr, out_affine)
    in_coords = [(0.0, dim - 1.0) for dim in shape]
    corners = np.asarray(np.meshgrid(*in_coords, indexing="ij")).reshape((len(shape), -1))
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    corners = in_affine @ corners
    corners_out = np.linalg.inv(out_affine) @ corners
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(corners_out.ptp(axis=1) + 1.0)
    if np.allclose(nib.io_orientation(in_affine), nib.io_orientation(out_affine)):
        # same orientation, get translate from the origin
        offset = in_affine @ ([0] * sr + [1])
        offset = offset[:-1] / offset[-1]
    else:
        # different orientation, the min is the origin
        corners = corners[:-1] / corners[-1]
        offset = np.min(corners, 1)
    return out_shape.astype(int), offset


def to_affine_nd(r: Union[np.ndarray, int], affine: np.ndarray) -> np.ndarray:
    """
    Using elements from affine, to create a new affine matrix by
    assigning the rotation/zoom/scaling matrix and the translation vector.

    when ``r`` is an integer, output is an (r+1)x(r+1) matrix,
    where the top left kxk elements are copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(r, len(affine) - 1)`.

    when ``r`` is an affine matrix, the output has the same as ``r``,
    the top left kxk elements are  copied from ``affine``,
    the last column of the output affine is copied from ``affine``'s last column.
    `k` is determined by `min(len(r) - 1, len(affine) - 1)`.

    Args:
        r (int or matrix): number of spatial dimensions or an output affine to be filled.
        affine (matrix): 2D affine matrix

    Raises:
        ValueError: When ``affine`` dimensions is not 2.
        ValueError: When ``r`` is nonpositive.

    Returns:
        an (r+1) x (r+1) matrix

    """
    affine_np = np.array(affine, dtype=np.float64)
    if affine_np.ndim != 2:
        raise ValueError(f"affine must have 2 dimensions, got {affine_np.ndim}.")
    new_affine = np.array(r, dtype=np.float64, copy=True)
    if new_affine.ndim == 0:
        sr = new_affine.astype(int)
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=np.float64)
    d = max(min(len(new_affine) - 1, len(affine_np) - 1), 1)
    new_affine[:d, :d] = affine_np[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_np[:d, -1]
    return new_affine


def create_file_basename(
    postfix: str,
    input_file_name: str,
    folder_path: str,
    data_root_dir: str = "",
) -> str:
    """
    Utility function to create the path to the output file based on the input
    filename (extension is added by lib level writer before writing the file)

    Args:
        postfix: output name's postfix
        input_file_name: path to the input image file.
        folder_path: path for the output file
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. This is used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names.
    """

    # get the filename and directory
    filedir, filename = os.path.split(input_file_name)
    # remove extension
    filename, ext = os.path.splitext(filename)
    if ext == ".gz":
        filename, ext = os.path.splitext(filename)
    # use data_root_dir to find relative path to file
    filedir_rel_path = ""
    if data_root_dir:
        filedir_rel_path = os.path.relpath(filedir, data_root_dir)

    # sub-folder path will be original name without the extension
    subfolder_path = os.path.join(folder_path, filedir_rel_path, filename)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # add the sub-folder plus the postfix name to become the file basename in the output path
    return os.path.join(subfolder_path, filename + "_" + postfix)


def compute_importance_map(
    patch_size: Tuple[int, ...],
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Get importance map for different weight modes.

    Args:
        patch_size: Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device: Device to put importance map on.

    Raises:
        ValueError: When ``mode`` is not one of ["constant", "gaussian"].

    Returns:
        Tensor of size patch_size.

    """
    mode = BlendMode(mode)
    if mode == BlendMode.CONSTANT:
        importance_map = torch.ones(patch_size, device=device).float()
    elif mode == BlendMode.GAUSSIAN:
        center_coords = [i // 2 for i in patch_size]
        sigma_scale = ensure_tuple_rep(sigma_scale, len(patch_size))
        sigmas = [i * sigma_s for i, sigma_s in zip(patch_size, sigma_scale)]

        importance_map = torch.zeros(patch_size, device=device)
        importance_map[tuple(center_coords)] = 1
        pt_gaussian = GaussianFilter(len(patch_size), sigmas).to(device=device, dtype=torch.float)
        importance_map = pt_gaussian(importance_map.unsqueeze(0).unsqueeze(0))
        importance_map = importance_map.squeeze(0).squeeze(0)
        importance_map = importance_map / torch.max(importance_map)
        importance_map = importance_map.float()

        # importance_map cannot be 0, otherwise we may end up with nans!
        min_non_zero = importance_map[importance_map != 0].min().item()
        importance_map = torch.clamp(importance_map, min=min_non_zero)
    else:
        raise ValueError(
            f"Unsupported mode: {mode}, available options are [{BlendMode.CONSTANT}, {BlendMode.CONSTANT}]."
        )

    return importance_map


def is_supported_format(filename: Union[Sequence[str], str], suffixes: Sequence[str]) -> bool:
    """
    Verify whether the specified file or files format match supported suffixes.
    If supported suffixes is None, skip the verification and return True.

    Args:
        filename: file name or a list of file names to read.
            if a list of files, verify all the suffixes.
        suffixes: all the supported image suffixes of current reader, must be a list of lower case suffixes.

    """
    filenames: Sequence[str] = ensure_tuple(filename)
    for name in filenames:
        tokens: Sequence[str] = PurePath(name).suffixes
        if len(tokens) == 0 or not any(("." + s.lower()) in "".join(tokens) for s in suffixes):
            return False

    return True
