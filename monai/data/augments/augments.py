"""
This contains the definitions of the commonly used argumentation functions. These apply operations to single instances
of data objects, which are tuples of numpy arrays where the first dimension if the channel dimension and others are
component, height/width (CHW), or height/width/depth (CHWD).
"""
from functools import partial

import numpy as np
import scipy.fftpack as ft
import scipy.ndimage

from monai.data.augments.decorators import augment, check_segment_margin
from monai.utils.arrayutils import (copypaste_arrays, rand_choice, rescale_array, resize_center)
from monai.utils.convutils import one_hot

try:
    from PIL import Image

    PILAvailable = True
except ImportError:
    PILAvailable = False


@augment()
def transpose(*arrs):
    """Transpose axes 1 and 2 for each of `arrs'."""
    return partial(np.swapaxes, axis1=1, axis2=2)


@augment()
def flip(*arrs):
    """Flip each of `arrs' with a random choice of up-down or left-right."""

    def _flip(arr):
        return arr[:, :, ::-1] if rand_choice() else arr[:, ::-1]

    return _flip


@augment()
def rot90(*arrs):
    """Rotate each of `arrs' a random choice of quarter, half, or three-quarter circle rotations."""
    return partial(np.rot90, k=np.random.randint(1, 3), axes=(1, 2))


@augment(prob=1.0)
def normalize(*arrs):
    """Normalize each of `arrs'."""
    return rescale_array


@augment(prob=1.0)
def rand_patch(*arrs, patch_size=(32, 32)):
    """Randomly choose a patch from `arrs' of dimensions `patch_size'."""
    ph, pw = patch_size

    def _rand_patch(im):
        h, w = im.shape[1:3]
        ry = np.random.randint(0, h - ph)
        rx = np.random.randint(0, w - pw)

        return im[:, ry:ry + ph, rx:rx + pw]

    return _rand_patch


@augment()
@check_segment_margin
def shift(*arrs, dim_fract=2, order=3):
    """Shift arrays randomly by `dimfract' fractions of the array dimensions."""
    testim = arrs[0]
    x, y = testim.shape[1:3]
    shiftx = np.random.randint(-x // dim_fract, x // dim_fract)
    shifty = np.random.randint(-y // dim_fract, y // dim_fract)

    def _shift(im):
        c, h, w = im.shape[:3]
        dest = np.zeros_like(im)

        srcslices, destslices = copypaste_arrays(im, dest, (0, h // 2 + shiftx, w // 2 + shifty), (0, h // 2, w // 2),
                                                 (c, h, w))
        dest[destslices] = im[srcslices]

        return dest

    return _shift


@augment()
@check_segment_margin
def rotate(*arrs):
    """Shift arrays randomly around the array center."""

    angle = np.random.random() * 360

    def _rotate(im):
        return scipy.ndimage.rotate(im, angle=angle, reshape=False, axes=(1, 2))

    return _rotate


@augment()
@check_segment_margin
def zoom(*arrs, zoomrange=0.2):
    """Return the image/mask pair zoomed by a random amount with the mask kept within `margin' pixels of the edges."""

    z = zoomrange - np.random.random() * zoomrange * 2
    zx = z + 1.0 + zoomrange * 0.25 - np.random.random() * zoomrange * 0.5
    zy = z + 1.0 + zoomrange * 0.25 - np.random.random() * zoomrange * 0.5

    def _zoom(im):
        ztemp = scipy.ndimage.zoom(im, (0, zx, zy) + tuple(1 for _ in range(1, im.ndim)), order=2)
        return resize_center(ztemp, *im.shape)

    return _zoom


@augment()
@check_segment_margin
def rotate_zoom_pil(*arrs, margin=5, min_fract=0.5, max_fract=2, resample=0):
    assert all(a.ndim >= 2 for a in arrs)
    assert PILAvailable, "PIL (pillow) not installed"

    testim = arrs[0]
    x, y = testim.shape[1:3]

    angle = np.random.random() * 360
    zoomx = x + np.random.randint(-x * min_fract, x * max_fract)
    zoomy = y + np.random.randint(-y * min_fract, y * max_fract)

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
def deform_pil(*arrs, defrange=25, num_controls=3, margin=2, map_order=1):
    """Deforms arrays randomly with a deformation grid of size `num_controls'**2 with `margins' grid values fixed."""
    assert PILAvailable, "PIL (pillow) not installed"

    h, w = arrs[0].shape[1:3]

    imshift = np.zeros((2, num_controls + margin * 2, num_controls + margin * 2))
    imshift[:, margin:-margin, margin:-margin] = np.random.randint(-defrange, defrange, (2, num_controls, num_controls))

    imshiftx = np.array(Image.fromarray(imshift[0]).resize((w, h), Image.QUAD))
    imshifty = np.array(Image.fromarray(imshift[1]).resize((w, h), Image.QUAD))

    y, x = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(x + imshiftx, (-1, 1)), np.reshape(y + imshifty, (-1, 1))

    def _map_channels(im):
        if im.ndim > 2:
            return np.stack(list(map(_map_channels, im)))
        elif im.ndim == 2:
            result = scipy.ndimage.map_coordinates(im, indices, order=map_order, mode="constant")
            return result.reshape(im.shape)

        raise ValueError("Incorrect image shape: %r" % (im.shape,))

    return _map_channels


@augment()
def distort_fft(*arrs, min_dist=0.1, max_dist=1.0):
    """Distorts arrays by applying dropout in k-space with a per-pixel probability based on distance from center."""
    h, w = arrs[0].shape[:2]

    x, y = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
    probfield = np.sqrt(x**2 + y**2)

    if arrs[0].ndim == 3:
        probfield = np.repeat(probfield[..., np.newaxis], arrs[0].shape[2], 2)

    dropout = np.random.uniform(min_dist, max_dist, arrs[0].shape) > probfield

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


def split_segmentation(*arrs, num_labels=2, seg_index=-1):
    arrs = list(arrs)
    seg = arrs[seg_index]
    seg = one_hot(seg, num_labels)
    arrs[seg_index] = seg

    return tuple(arrs)


def merge_segmentation(*arrs, seg_index=-1):
    arrs = list(arrs)
    seg = arrs[seg_index]
    seg = np.argmax(seg, 2)
    arrs[seg_index] = seg

    return tuple(arrs)
