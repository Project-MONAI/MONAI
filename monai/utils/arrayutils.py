
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
import numpy as np


def randChoice(prob=0.5):
    """Returns True if a randomly chosen number is less than or equal to `prob', by default this is a 50/50 chance."""
    return random.random() <= prob


def imgBounds(img):
    """Returns the minimum and maximum indices of non-zero lines in axis 0 of `img', followed by that for axis 1."""
    ax0 = np.any(img, axis=0)
    ax1 = np.any(img, axis=1)
    return np.concatenate((np.where(ax0)[0][[0, -1]], np.where(ax1)[0][[0, -1]]))


def inBounds(x, y, margin, maxx, maxy):
    """Returns True if (x,y) is within the rectangle (margin,margin,maxx-margin,maxy-margin)."""
    return margin <= x < (maxx - margin) and margin <= y < (maxy - margin)


def isEmpty(img):
    """Returns True if `img' is empty, that is its maximum value is not greater than its minimum."""
    return not (img.max() > img.min())  # use > instead of <= so that an image full of NaNs will result in True


def ensureTupleSize(tup, dim):
    """Returns a copy of `tup' with `dim' values by either shortened or padded with zeros as necessary."""
    tup = tuple(tup) + (0,) * dim
    return tup[:dim]


def zeroMargins(img, margin):
    """Returns True if the values within `margin' indices of the edges of `img' in dimensions 1 and 2 are 0."""
    if np.any(img[:, :, :margin]) or np.any(img[:, :, -margin:]):
        return False

    if np.any(img[:, :margin, :]) or np.any(img[:, -margin:, :]):
        return False

    return True


def rescaleArray(arr, minv=0.0, maxv=1.0, dtype=np.float32):
    """Rescale the values of numpy array `arr' to be from `minv' to `maxv'."""
    if dtype is not None:
        arr = arr.astype(dtype)

    mina = np.min(arr)
    maxa = np.max(arr)

    if mina == maxa:
        return arr * minv

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default


def rescaleInstanceArray(arr, minv=0.0, maxv=1.0, dtype=np.float32):
    """Rescale each array slice along the first dimension of `arr' independently."""
    out = np.zeros(arr.shape, dtype)
    for i in range(arr.shape[0]):
        out[i] = rescaleArray(arr[i], minv, maxv, dtype)

    return out


def rescaleArrayIntMax(arr, dtype=np.uint16):
    """Rescale the array `arr' to be between the minimum and maximum values of the type `dtype'."""
    info = np.iinfo(dtype)
    return rescaleArray(arr, info.min, info.max).astype(dtype)


def copypasteArrays(src, dest, srccenter, destcenter, dims):
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
            d1 = np.clip(dim // 2, 0, min(sc, dc))  # dimension before midpoint, clip to size fitting in both arrays
            d2 = np.clip(dim // 2 + 1, 0, min(ss - sc, ds - dc))  # dimension after midpoint, clip to size fitting in both arrays

            srcslices[i] = slice(sc - d1, sc + d2)
            destslices[i] = slice(dc - d1, dc + d2)

    return tuple(srcslices), tuple(destslices)


def resizeCenter(img, *resizeDims, fillValue=0):
    """
    Resize `img' by cropping or expanding the image from the center. The `resizeDims' values are the output dimensions
    (or None to use original dimension of `img'). If a dimension is smaller than that of `img' then the result will be
    cropped and if larger padded with zeros, in both cases this is done relative to the center of `img'. The result is
    a new image with the specified dimensions and values from `img' copied into its center.
    """
    resizeDims = tuple(resizeDims[i] or img.shape[i] for i in range(len(resizeDims)))

    dest = np.full(resizeDims, fillValue, img.dtype)
    srcslices, destslices = copypasteArrays(img, dest, np.asarray(img.shape) // 2, np.asarray(dest.shape) // 2, resizeDims)
    dest[destslices] = img[srcslices]

    return dest
