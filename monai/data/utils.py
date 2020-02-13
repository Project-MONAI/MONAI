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
from itertools import starmap, product

import numpy as np
from monai.transforms.utils import ensure_tuple_size


def get_random_patch(dims, patch_size):
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size` or the as
    close to it as possible within the given dimension. It is expected that `patch_size` is a valid patch for a source
    of shape `dims` as returned by `get_valid_patch_size`.

    Args:
        dims (tuple of int): shape of source array
        patch_size (tuple of int): shape of patch size to generate

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """

    # choose the minimal corner of the patch
    min_corner = tuple(np.random.randint(0, ms - ps) if ms > ps else 0 for ms, ps in zip(dims, patch_size))

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
    Yield successive patches from `arr' of size `patchSize'. The iteration can start from position `startPos' in `arr'
    but drawing from a padded array extended by the `patchSize' in each dimension (so these coordinates can be negative
    to start in the padded region). If `copyBack' is True the values from each patch are written back to `arr'.

    Args:
        arr (np.ndarray): array to iterate over
        patch_size (tuple of int or None): size of patches to generate slices for, 0 or None selects whole dimension
        start_pos (tuple of it, optional): starting position in the array, default is 0 for each dimension
        copy_back (bool): if True data from the yielded patches is copied back to `arr` once the generator completes
        pad_mode (str, optional): padding mode, see numpy.pad
        pad_opts (dict, optional): padding options, see numpy.pad

    Yields:
        Patches of array data from `arr` which are views into a padded array which can be modified, if `copy_back` is
        True these changes will be reflected in `arr` once the iteration completes
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
