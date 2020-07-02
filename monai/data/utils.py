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

from typing import Optional, Union

import os
import warnings
import math
from itertools import starmap, product
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
from monai.utils import ensure_tuple_size, ensure_tuple_rep, optional_import, NumpyPadMode, BlendMode
from monai.networks.layers.simplelayers import GaussianFilter

nib, _ = optional_import("nibabel")


def get_random_patch(dims, patch_size, rand_state: Optional[np.random.RandomState] = None):
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
    close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
    of shape `dims` as returned by `get_valid_patch_size`.

    Args:
        dims (tuple of int): shape of source array
        patch_size (tuple of int): shape of patch size to generate
        rand_state (np.random.RandomState): a random state object to generate random numbers from

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """

    # choose the minimal corner of the patch
    rand_int = np.random.randint if rand_state is None else rand_state.randint
    min_corner = tuple(rand_int(0, ms - ps) if ms > ps else 0 for ms, ps in zip(dims, patch_size))

    # create the slices for each dimension which define the patch in the source array
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))


def iter_patch_slices(dims, patch_size, start_pos=()):
    """
    Yield successive tuples of slices defining patches of size `patch_size` from an array of dimensions `dims`. The
    iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
    patch is chosen in a contiguous grid using a first dimension as least significant ordering.

    Args:
        dims (tuple of int): dimensions of array to iterate over
        patch_size (tuple of int or None): size of patches to generate slices for, 0 or None selects whole dimension
        start_pos (tuple of it, optional): starting position in the array, default is 0 for each dimension

    Yields:
        Tuples of slice objects defining each patch
    """

    # ensure patchSize and startPos are the right length
    ndim = len(dims)
    patch_size = get_valid_patch_size(dims, patch_size)
    start_pos = ensure_tuple_size(start_pos, ndim)

    # collect the ranges to step over each dimension
    ranges = tuple(starmap(range, zip(start_pos, dims, patch_size)))

    # choose patches by applying product to the ranges
    for position in product(*ranges[::-1]):  # reverse ranges order to iterate in index order
        yield tuple(slice(s, s + p) for s, p in zip(position[::-1], patch_size))


def dense_patch_slices(image_size, patch_size, scan_interval):
    """
    Enumerate all slices defining 2D/3D patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size (tuple of int): dimensions of image to iterate over
        patch_size (tuple of int): size of patches to generate slices
        scan_interval (tuple of int): dense patch sampling interval

    Returns:
        a list of slice objects defining each patch

    Raises:
        ValueError: image_size should have 2 or 3 elements

    """
    num_spatial_dims = len(image_size)
    if num_spatial_dims not in (2, 3):
        raise ValueError("image_size should have 2 or 3 elements")
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = [
        int(math.ceil(float(image_size[i]) / scan_interval[i])) if scan_interval[i] != 0 else 1
        for i in range(num_spatial_dims)
    ]
    slices = []
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
    patch_size,
    start_pos=(),
    copy_back: bool = True,
    mode: Union[NumpyPadMode, str] = NumpyPadMode.WRAP,
    **pad_opts,
):
    """
    Yield successive patches from `arr` of size `patch_size`. The iteration can start from position `start_pos` in `arr`
    but drawing from a padded array extended by the `patch_size` in each dimension (so these coordinates can be negative
    to start in the padded region). If `copy_back` is True the values from each patch are written back to `arr`.

    Args:
        arr (np.ndarray): array to iterate over
        patch_size (tuple of int or None): size of patches to generate slices for, 0 or None selects whole dimension
        start_pos (tuple of it, optional): starting position in the array, default is 0 for each dimension
        copy_back: if True data from the yielded patches is copied back to `arr` once the generator completes
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
        pad_opts (dict, optional): padding options, see `numpy.pad`

    Yields:
        Patches of array data from `arr` which are views into a padded array which can be modified, if `copy_back` is
        True these changes will be reflected in `arr` once the iteration completes.
    """
    # ensure patchSize and startPos are the right length
    patch_size = get_valid_patch_size(arr.shape, patch_size)
    start_pos = ensure_tuple_size(start_pos, arr.ndim)

    # pad image by maximum values needed to ensure patches are taken from inside an image
    arrpad = np.pad(arr, tuple((p, p) for p in patch_size), NumpyPadMode(mode).value, **pad_opts)

    # choose a start position in the padded image
    start_pos_padded = tuple(s + p for s, p in zip(start_pos, patch_size))

    # choose a size to iterate over which is smaller than the actual padded image to prevent producing
    # patches which are only in the padded regions
    iter_size = tuple(s + p for s, p in zip(arr.shape, patch_size))

    for slices in iter_patch_slices(iter_size, patch_size, start_pos_padded):
        yield arrpad[slices]

    # copy back data from the padded image if required
    if copy_back:
        slices = tuple(slice(p, p + s) for p, s in zip(patch_size, arr.shape))
        arr[...] = arrpad[slices]


def get_valid_patch_size(image_size, patch_size):
    """
    Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
    the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `image_size` with that size in each dimension.
    """
    ndim = len(image_size)

    try:
        # if a single value was given as patch size, treat this as the size of the patch over all dimensions
        single_patch_size = int(patch_size)
        patch_size = ensure_tuple_rep(single_patch_size, ndim)
    except TypeError:  # raised if the patch size is multiple values
        # ensure patch size is at least as long as number of dimensions
        patch_size = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size))


def list_data_collate(batch):
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


def worker_init_fn(worker_id):
    """
    Callback function for PyTorch DataLoader `worker_init_fn`.
    It can set different random seed for the transforms in different workers.

    """
    worker_info = torch.utils.data.get_worker_info()  # type: ignore
    if hasattr(worker_info.dataset, "transform") and hasattr(worker_info.dataset.transform, "set_random_state"):
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))


def correct_nifti_header_if_necessary(img_nii):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.

    Args:
        img_nii (nifti image object)
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


def zoom_affine(affine, scale, diagonal: bool = True):
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
        scale (sequence of floats): new scaling factor along each dimension.
        diagonal: whether to return a diagonal scaling matrix.
            Defaults to True.

    Returns:
        the updated `n x n` affine.

    Raises:
        ValueError: affine should be a square matrix
        ValueError: scale must be a sequence of positive numbers.

    """
    affine = np.array(affine, dtype=float, copy=True)
    if len(affine) != len(affine[0]):
        raise ValueError("affine should be a square matrix")
    scale = np.array(scale, dtype=float, copy=True)
    if np.any(scale <= 0):
        raise ValueError("scale must be a sequence of positive numbers.")
    d = len(affine) - 1
    if len(scale) < d:  # defaults based on affine
        norm = np.sqrt(np.sum(np.square(affine), 0))[:-1]
        scale = np.append(scale, norm[len(scale) :])
    scale = scale[:d]
    scale[scale == 0] = 1.0
    if diagonal:
        return np.diag(np.append(scale, [1.0]))
    rzs = affine[:-1, :-1]  # rotation zoom scale
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    s = np.sign(np.diag(zs)) * np.abs(scale)
    # construct new affine with rotation and zoom
    new_affine = np.eye(len(affine))
    new_affine[:-1, :-1] = rotation @ np.diag(s)
    return new_affine


def compute_shape_offset(spatial_shape, in_affine, out_affine):
    """
    Given input and output affine, compute appropriate shapes
    in the output space based on the input array's shape.
    This function also returns the offset to put the shape
    in a good position with respect to the world coordinate system.
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


def to_affine_nd(r, affine):
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

    Returns:
        an (r+1) x (r+1) matrix

    Raises:
        ValueError: input affine matrix must have two dimensions, got {affine.ndim}.
        ValueError: r must be positive, got {sr}.

    """
    affine_ = np.array(affine, dtype=np.float64)
    if affine_.ndim != 2:
        raise ValueError(f"input affine matrix must have two dimensions, got {affine_.ndim}.")
    new_affine = np.array(r, dtype=np.float64, copy=True)
    if new_affine.ndim == 0:
        sr = new_affine.astype(int)
        if not np.isfinite(sr) or sr < 0:
            raise ValueError(f"r must be positive, got {sr}.")
        new_affine = np.eye(sr + 1, dtype=np.float64)
    d = max(min(len(new_affine) - 1, len(affine_) - 1), 1)
    new_affine[:d, :d] = affine_[:d, :d]
    if d > 1:
        new_affine[:d, -1] = affine_[:d, -1]
    return new_affine


def create_file_basename(postfix: str, input_file_name: str, folder_path: str, data_root_dir: str = ""):
    """
    Utility function to create the path to the output file based on the input
    filename (extension is added by lib level writer before writing the file)

    Args:
        postfix: output name's postfix
        input_file_name: path to the input image file
        folder_path: path for the output file
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. This is used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names.
    """

    # get the filename and directory
    filedir, filename = os.path.split(input_file_name)

    # jettison the extension to have just filename
    filename, ext = os.path.splitext(filename)
    while ext != "":
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
    patch_size, mode: Union[BlendMode, str] = BlendMode.CONSTANT, sigma_scale: float = 0.125, device=None
):
    """Get importance map for different weight modes.

    Args:
        patch_size (tuple): Size of the required importance map. This should be either H, W [,D].
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: Sigma_scale to calculate sigma for each dimension
            (sigma = sigma_scale * dim_size). Used for gaussian mode only.
        device (str of pytorch device): Device to put importance map on.

    Returns:
        Tensor of size patch_size.

    Raises:
        ValueError: mode must be "constant" or "gaussian".

    """
    mode = BlendMode(mode)
    if mode == BlendMode.CONSTANT:
        importance_map = torch.ones(patch_size, device=device).float()
    elif mode == BlendMode.GAUSSIAN:
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]

        importance_map = torch.zeros(patch_size, device=device)
        importance_map[tuple(center_coords)] = 1
        pt_gaussian = GaussianFilter(len(patch_size), sigmas).to(device=device, dtype=torch.float)
        importance_map = pt_gaussian(importance_map.unsqueeze(0).unsqueeze(0))
        importance_map = importance_map.squeeze(0).squeeze(0)
        importance_map = importance_map / torch.max(importance_map)
        importance_map = importance_map.float()

        # importance_map cannot be 0, otherwise we may end up with nans!
        importance_map[importance_map == 0] = torch.min(importance_map[importance_map != 0])
    else:
        raise ValueError('mode must be "constant" or "gaussian".')

    return importance_map
