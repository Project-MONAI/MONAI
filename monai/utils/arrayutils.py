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


import random
from itertools import product, starmap

import numpy as np


def rand_choice(prob=0.5):
    """Returns True if a randomly chosen number is less than or equal to `prob', by default this is a 50/50 chance."""
    return random.random() <= prob


def img_bounds(img):
    """Returns the minimum and maximum indices of non-zero lines in axis 0 of `img', followed by that for axis 1."""
    ax0 = np.any(img, axis=0)
    ax1 = np.any(img, axis=1)
    return np.concatenate((np.where(ax0)[0][[0, -1]], np.where(ax1)[0][[0, -1]]))


def in_bounds(x, y, margin, maxx, maxy):
    """Returns True if (x,y) is within the rectangle (margin,margin,maxx-margin,maxy-margin)."""
    return margin <= x < (maxx - margin) and margin <= y < (maxy - margin)


def is_empty(img):
    """Returns True if `img' is empty, that is its maximum value is not greater than its minimum."""
    return not (img.max() > img.min())  # use > instead of <= so that an image full of NaNs will result in True


def ensure_tuple_size(tup, dim):
    """Returns a copy of `tup' with `dim' values by either shortened or padded with zeros as necessary."""
    tup = tuple(tup) + (0,) * dim
    return tup[:dim]


def zero_margins(img, margin):
    """Returns True if the values within `margin' indices of the edges of `img' in dimensions 1 and 2 are 0."""
    if np.any(img[:, :, :margin]) or np.any(img[:, :, -margin:]):
        return False

    if np.any(img[:, :margin, :]) or np.any(img[:, -margin:, :]):
        return False

    return True


def rescale_array(arr, minv=0.0, maxv=1.0, dtype=np.float32):
    """Rescale the values of numpy array `arr' to be from `minv' to `maxv'."""
    if dtype is not None:
        arr = arr.astype(dtype)

    mina = np.min(arr)
    maxa = np.max(arr)

    if mina == maxa:
        return arr * minv

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default


def rescale_instance_array(arr, minv=0.0, maxv=1.0, dtype=np.float32):
    """Rescale each array slice along the first dimension of `arr' independently."""
    out = np.zeros(arr.shape, dtype)
    for i in range(arr.shape[0]):
        out[i] = rescale_array(arr[i], minv, maxv, dtype)

    return out


def rescale_array_int_max(arr, dtype=np.uint16):
    """Rescale the array `arr' to be between the minimum and maximum values of the type `dtype'."""
    info = np.iinfo(dtype)
    return rescale_array(arr, info.min, info.max).astype(dtype)


def copypaste_arrays(src, dest, srccenter, destcenter, dims):
    """
    Calculate the slices to copy a sliced area of array `src' into array `dest'. The area has dimensions `dims' (use 0
    or None to copy everything in that dimension), the source area is centered at `srccenter' index in `src' and copied
    into area centered at `destcenter' in `dest'. The dimensions of the copied area will be clipped to fit within the
    source and destination arrays so a smaller area may be copied than expected. Return value is the tuples of slice
    objects indexing the copied area in `src', and those indexing the copy area in `dest'.

    Example:
        src=np.random.randint(0,10,(6,6))
        dest=np.zeros_like(src)
        srcslices,destslices=copypasteArrays(src,dest,(3,2),(2,1),(3,4))
        dest[destslices]=src[srcslices]
        print(src)
        print(dest)

        >>> [[9 5 6 6 9 6]
             [4 3 5 6 1 2]
             [0 7 3 2 4 1]
             [3 0 0 1 5 1]
             [9 4 7 1 8 2]
             [6 6 5 8 6 7]]
            [[0 0 0 0 0 0]
             [7 3 2 4 0 0]
             [0 0 1 5 0 0]
             [4 7 1 8 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]]
    """
    srcslices = [slice(None)] * src.ndim
    destslices = [slice(None)] * dest.ndim

    for i, ss, ds, sc, dc, dim in zip(range(src.ndim), src.shape, dest.shape, srccenter, destcenter, dims):
        if dim:
            # dimension before midpoint, clip to size fitting in both arrays
            d1 = np.clip(dim // 2, 0, min(sc, dc))
            # dimension after midpoint, clip to size fitting in both arrays
            d2 = np.clip(dim // 2 + 1, 0, min(ss - sc, ds - dc))

            srcslices[i] = slice(sc - d1, sc + d2)
            destslices[i] = slice(dc - d1, dc + d2)

    return tuple(srcslices), tuple(destslices)


def resize_center(img, *resize_dims, fill_value=0):
    """
    Resize `img' by cropping or expanding the image from the center. The `resizeDims' values are the output dimensions
    (or None to use original dimension of `img'). If a dimension is smaller than that of `img' then the result will be
    cropped and if larger padded with zeros, in both cases this is done relative to the center of `img'. The result is
    a new image with the specified dimensions and values from `img' copied into its center.
    """
    resize_dims = tuple(resize_dims[i] or img.shape[i] for i in range(len(resize_dims)))

    dest = np.full(resize_dims, fill_value, img.dtype)
    half_img_shape = np.asarray(img.shape) // 2
    half_dest_shape = np.asarray(dest.shape) // 2

    srcslices, destslices = copypaste_arrays(img, dest, half_img_shape, half_dest_shape, resize_dims)
    dest[destslices] = img[srcslices]

    return dest


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
