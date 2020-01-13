
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


"""
This contains the definitions of the commonly used argumentation functions. These apply operations to single instances
of data objects, which are tuples of numpy arrays where the first dimension if the channel dimension and others are 
component, height/width (CHW), or height/width/depth (CHWD).
"""
from functools import partial
import numpy as np
import scipy.ndimage
import scipy.fftpack as ft

from monai.data.augments.decorators import augment, checkSegmentMargin
from monai.utils.arrayutils import randChoice, rescaleArray, copypasteArrays, resizeCenter
from monai.utils.convutils import oneHot

try:
    from PIL import Image

    pilAvailable = True
except ImportError:
    pilAvailable = False


@augment()
def transpose(*arrs):
    """Transpose axes 1 and 2 for each of `arrs'."""
    return partial(np.swapaxes, axis1=1, axis2=2)


@augment()
def flip(*arrs):
    """Flip each of `arrs' with a random choice of up-down or left-right."""

    def _flip(arr):
        return arr[:, :, ::-1] if randChoice() else arr[:, ::-1]

    return _flip


@augment()
def rot90(*arrs):
    """Rotate each of `arrs' a random choice of quarter, half, or three-quarter circle rotations."""
    return partial(np.rot90, k=np.random.randint(1, 3), axes=(1, 2))


@augment(prob=1.0)
def normalize(*arrs):
    """Normalize each of `arrs'."""
    return rescaleArray


@augment(prob=1.0)
def randPatch(*arrs, patchSize=(32, 32)):
    """Randomly choose a patch from `arrs' of dimensions `patchSize'."""
    ph, pw = patchSize

    def _randPatch(im):
        h, w = im.shape[1:3]
        ry = np.random.randint(0, h - ph)
        rx = np.random.randint(0, w - pw)

        return im[:, ry : ry + ph, rx : rx + pw]

    return _randPatch


@augment()
@checkSegmentMargin
def shift(*arrs, dimFract=2, order=3):
    """Shift arrays randomly by `dimfract' fractions of the array dimensions."""
    testim = arrs[0]
    x, y = testim.shape[1:3]
    shiftx = np.random.randint(-x // dimFract, x // dimFract)
    shifty = np.random.randint(-y // dimFract, y // dimFract)

    def _shift(im):
        c, h, w = im.shape[:3]
        dest = np.zeros_like(im)

        srcslices, destslices = copypasteArrays(im, dest, (0, h // 2 + shiftx, w // 2 + shifty), 
                                                (0, h // 2, w // 2), (c, h, w))
        dest[destslices] = im[srcslices]

        return dest

    return _shift


@augment()
@checkSegmentMargin
def rotate(*arrs):
    """Shift arrays randomly around the array center."""

    angle = np.random.random() * 360

    def _rotate(im):
        return scipy.ndimage.rotate(im, angle=angle, reshape=False, axes=(1, 2))

    return _rotate


@augment()
@checkSegmentMargin
def zoom(*arrs, zoomrange=0.2):
    """Return the image/mask pair zoomed by a random amount with the mask kept within `margin' pixels of the edges."""

    z = zoomrange - np.random.random() * zoomrange * 2
    zx = z + 1.0 + zoomrange * 0.25 - np.random.random() * zoomrange * 0.5
    zy = z + 1.0 + zoomrange * 0.25 - np.random.random() * zoomrange * 0.5

    def _zoom(im):
        ztemp = scipy.ndimage.zoom(im, (0, zx, zy) + tuple(1 for _ in range(1, im.ndim)), order=2)
        return resizeCenter(ztemp, *im.shape)

    return _zoom


@augment()
@checkSegmentMargin
def rotateZoomPIL(*arrs, margin=5, minFract=0.5, maxFract=2, resample=0):
    assert all(a.ndim >= 2 for a in arrs)
    assert pilAvailable, "PIL (pillow) not installed"

    testim = arrs[0]
    x, y = testim.shape[1:3]

    angle = np.random.random() * 360
    zoomx = x + np.random.randint(-x * minFract, x * maxFract)
    zoomy = y + np.random.randint(-y * minFract, y * maxFract)

    filters = (Image.NEAREST, Image.LINEAR, Image.BICUBIC)

    def _trans(im):
        if im.dtype != np.float32:
            return _trans(im.astype(np.float32)).astype(im.dtype)
        if im.ndim > 2:
            return np.stack(list(map(_trans, im)))
        elif im.ndim == 2:
            im = Image.fromarray(im)

            # rotation
            im = im.rotate(angle, filters[resample])

            # zoom
            zoomsize = (zoomx, zoomy)
            pastesize = (im.size[0] // 2 - zoomsize[0] // 2, im.size[1] // 2 - zoomsize[1] // 2)
            newim = Image.new("F", im.size)
            newim.paste(im.resize(zoomsize, filters[resample]), pastesize)
            im = newim

            return np.array(im)

        raise ValueError("Incorrect image shape: %r" % (im.shape,))

    return _trans


@augment()
def deformPIL(*arrs, defrange=25, numControls=3, margin=2, mapOrder=1):
    """Deforms arrays randomly with a deformation grid of size `numControls'**2 with `margins' grid values fixed."""
    assert pilAvailable, "PIL (pillow) not installed"

    h, w = arrs[0].shape[1:3]

    imshift = np.zeros((2, numControls + margin * 2, numControls + margin * 2))
    imshift[:, margin:-margin, margin:-margin] = np.random.randint(-defrange, defrange, (2, numControls, numControls))

    imshiftx = np.array(Image.fromarray(imshift[0]).resize((w, h), Image.QUAD))
    imshifty = np.array(Image.fromarray(imshift[1]).resize((w, h), Image.QUAD))

    y, x = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(x + imshiftx, (-1, 1)), np.reshape(y + imshifty, (-1, 1))

    def _mapChannels(im):
        if im.ndim > 2:
            return np.stack(list(map(_mapChannels, im)))
        elif im.ndim == 2:
            result = scipy.ndimage.map_coordinates(im, indices, order=mapOrder, mode="constant")
            return result.reshape(im.shape)

        raise ValueError("Incorrect image shape: %r" % (im.shape,))

    return _mapChannels


@augment()
def distortFFT(*arrs, minDist=0.1, maxDist=1.0):
    """Distorts arrays by applying dropout in k-space with a per-pixel probability based on distance from center."""
    h, w = arrs[0].shape[:2]

    x, y = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
    probfield = np.sqrt(x ** 2 + y ** 2)

    if arrs[0].ndim == 3:
        probfield = np.repeat(probfield[..., np.newaxis], arrs[0].shape[2], 2)

    dropout = np.random.uniform(minDist, maxDist, arrs[0].shape) > probfield

    def _distort(im):
        if im.ndim == 2:
            result = ft.fft2(im)
            result = ft.fftshift(result)
            result = result * dropout[:, :, 0]
            result = ft.ifft2(result)
            result = np.abs(result)
        else:
            result = np.dstack([_distort(im[..., i]) for i in range(im.shape[-1])])

        return result

    return _distort


def splitSegmentation(*arrs, numLabels=2, segIndex=-1):
    arrs = list(arrs)
    seg = arrs[segIndex]
    seg = oneHot(seg, numLabels)
    arrs[segIndex] = seg

    return tuple(arrs)


def mergeSegmentation(*arrs, segIndex=-1):
    arrs = list(arrs)
    seg = arrs[segIndex]
    seg = np.argmax(seg, 2)
    arrs[segIndex] = seg

    return tuple(arrs)
