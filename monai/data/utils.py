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

import warnings
import math
from itertools import starmap, product
from torch.utils.data._utils.collate import default_collate
import numpy as np
from monai.transforms.utils import ensure_tuple_size


def get_random_patch(dims, patch_size, rand_state=None):
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
    """
    num_spatial_dims = len(image_size)
    if num_spatial_dims not in (2, 3):
        raise ValueError('image_size should has 2 or 3 elements')
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = [int(math.ceil(float(image_size[i]) / scan_interval[i])) if scan_interval[i] != 0 else 1
                for i in range(num_spatial_dims)]
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


def iter_patch(arr, patch_size, start_pos=(), copy_back=True, pad_mode="wrap", **pad_opts):
    """
    Yield successive patches from `arr` of size `patch_size`. The iteration can start from position `start_pos` in `arr`
    but drawing from a padded array extended by the `patch_size` in each dimension (so these coordinates can be negative
    to start in the padded region). If `copy_back` is True the values from each patch are written back to `arr`.

    Args:
        arr (np.ndarray): array to iterate over
        patch_size (tuple of int or None): size of patches to generate slices for, 0 or None selects whole dimension
        start_pos (tuple of it, optional): starting position in the array, default is 0 for each dimension
        copy_back (bool): if True data from the yielded patches is copied back to `arr` once the generator completes
        pad_mode (str, optional): padding mode, see `numpy.pad`
        pad_opts (dict, optional): padding options, see `numpy.pad`

    Yields:
        Patches of array data from `arr` which are views into a padded array which can be modified, if `copy_back` is
        True these changes will be reflected in `arr` once the iteration completes.
    """
    # ensure patchSize and startPos are the right length
    patch_size = get_valid_patch_size(arr.shape, patch_size)
    start_pos = ensure_tuple_size(start_pos, arr.ndim)

    # pad image by maximum values needed to ensure patches are taken from inside an image
    arrpad = np.pad(arr, tuple((p, p) for p in patch_size), pad_mode, **pad_opts)

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


def get_valid_patch_size(dims, patch_size):
    """
    Given an image of dimensions `dims`, return a patch size tuple taking the dimension from `patch_size` if this is
    not 0/None. Otherwise, or if `patch_size` is shorter than `dims`, the dimension from `dims` is taken. This ensures
    the returned patch size is within the bounds of `dims`. If `patch_size` is a single number this is interpreted as a
    patch of the same dimensionality of `dims` with that size in each dimension.
    """
    ndim = len(dims)

    try:
        # if a single value was given as patch size, treat this as the size of the patch over all dimensions
        single_patch_size = int(patch_size)
        patch_size = (single_patch_size,) * ndim
    except TypeError:  # raised if the patch size is multiple values
        # ensure patch size is at least as long as number of dimensions
        patch_size = ensure_tuple_size(patch_size, ndim)

    # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
    return tuple(min(ms, ps or ms) for ms, ps in zip(dims, patch_size))


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


def correct_nifti_header_if_necessary(img_nii):
    """
    check nifti object header's format, update the header if needed.
    in the updated image pixdim matches the affine.

    Args:
        img (nifti image object)
    """
    dim = img_nii.header['dim'][0]
    if dim >= 5:
        return img_nii  # do nothing for high-dimensional array
    # check that affine matches zooms
    pixdim = np.asarray(img_nii.header.get_zooms())[:dim]
    norm_affine = np.sqrt(np.sum(np.square(img_nii.affine[:dim, :dim]), 0))
    if np.allclose(pixdim, norm_affine):
        return img_nii
    if hasattr(img_nii, 'get_sform'):
        return rectify_header_sform_qform(img_nii)
    return img_nii


def rectify_header_sform_qform(img_nii):
    """
    Look at the sform and qform of the nifti object and correct it if any
    incompatibilities with pixel dimensions

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/io/misc_io.py
    """
    d = img_nii.header['dim'][0]
    pixdim = np.asarray(img_nii.header.get_zooms())[:d]
    sform, qform = img_nii.get_sform(), img_nii.get_qform()
    norm_sform = np.sqrt(np.sum(np.square(sform[:d, :d]), 0))
    norm_qform = np.sqrt(np.sum(np.square(qform[:d, :d]), 0))
    sform_mismatch = not np.allclose(norm_sform, pixdim)
    qform_mismatch = not np.allclose(norm_qform, pixdim)

    if img_nii.header['sform_code'] != 0:
        if not sform_mismatch:
            return img_nii
        if not qform_mismatch:
            img_nii.set_sform(img_nii.get_qform())
            return img_nii
    if img_nii.header['qform_code'] != 0:
        if not qform_mismatch:
            return img_nii
        if not sform_mismatch:
            img_nii.set_qform(img_nii.get_sform())
            return img_nii

    norm_affine = np.sqrt(np.sum(np.square(img_nii.affine[:, :3]), 0))
    to_divide = np.tile(np.expand_dims(np.append(norm_affine, 1), axis=1), [1, 4])
    pixdim = np.append(pixdim, [1.] * (4 - len(pixdim)))
    to_multiply = np.tile(np.expand_dims(pixdim, axis=1), [1, 4])
    affine = img_nii.affine / to_divide.T * to_multiply.T
    warnings.warn('Modifying image affine from {} to {}'.format(img_nii.affine, affine))

    img_nii.set_sform(affine)
    img_nii.set_qform(affine)
    return img_nii
