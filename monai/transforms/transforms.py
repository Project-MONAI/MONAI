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
A collection of "vanilla" transforms
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import warnings
import numpy as np
import scipy.ndimage
import nibabel as nib
from PIL import Image
import torch
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from skimage.transform import resize

from monai.data.utils import (get_random_patch, get_valid_patch_size, correct_nifti_header_if_necessary, zoom_affine,
                              compute_shape_offset, to_affine_nd)
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.compose import Randomizable
from monai.transforms.utils import (create_control_grid, create_grid, create_rotate, create_scale, create_shear,
                                    create_translate, rescale_array)
from monai.utils.misc import ensure_tuple


class Spacing:
    """
    Resample input image into the specified `pixdim`.
    """

    def __init__(self, pixdim, diagonal=False, mode='constant', cval=0, dtype=None):
        """
        Args:
            pixdim (sequence of floats): output voxel spacing.
            diagonal (bool): whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data.
            mode (`reflect|constant|nearest|mirror|wrap`):
                The mode parameter determines how the input array is extended beyond its boundaries.
            cval (scalar): Value to fill past edges of input if mode is "constant". Default is 0.0.
            dtype (None or np.dtype): output array data type, defaults to None to use input data's dtype.
        """
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.mode = mode
        self.cval = cval
        self.dtype = dtype

    def __call__(self, data_array, original_affine=None, interp_order=3):
        """
        Args:
            data_array (ndarray): in shape (num_channels, H[, W, ...]).
            original_affine (4x4 matrix): original affine. Defaults to "eye(4)".
            interp_order (int): The order of the spline interpolation, default is 3.
                The order has to be in the range 0-5.
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
        Returns:
            data_array (resampled into `self.pixdim`), original pixdim, current pixdim.
        """
        sr = data_array.ndim - 1
        if sr <= 0:
            raise ValueError('the array should have at least one spatial dimension.')
        if original_affine is None:
            # default to identity
            original_affine = np.eye(sr + 1, dtype=np.float64)
            affine = np.eye(sr + 1, dtype=np.float64)
        else:
            affine = to_affine_nd(sr, original_affine)
        out_d = self.pixdim[:sr]
        if out_d.size < sr:
            out_d = np.append(out_d, [1.] * (out_d.size - sr))
        if np.any(out_d <= 0):
            raise ValueError('pixdim must be positive, got {}'.format(out_d))
        # compute output affine, shape and offset
        new_affine = zoom_affine(affine, out_d, diagonal=self.diagonal)
        output_shape, offset = compute_shape_offset(data_array.shape[1:], affine, new_affine)
        new_affine[:sr, -1] = offset[:sr]
        transform = np.linalg.inv(affine) @ new_affine
        # adapt to the actual rank
        transform_ = to_affine_nd(sr, transform)
        # resample
        dtype = data_array.dtype if self.dtype is None else self.dtype
        output_data = []
        for data in data_array:
            data_ = scipy.ndimage.affine_transform(
                data.astype(dtype), matrix=transform_, output_shape=output_shape,
                order=interp_order, mode=self.mode, cval=self.cval)
            output_data.append(data_)
        output_data = np.stack(output_data)
        new_affine = to_affine_nd(original_affine, new_affine)
        return output_data, original_affine, new_affine


class Orientation:
    """
    Change the input image's orientation into the specified based on `axcodes`.
    """

    def __init__(self, axcodes=None, as_closest_canonical=False, labels=tuple(zip('LPI', 'RAS'))):
        """
        Args:
            axcodes (N elements sequence): for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical (boo): if True, load the image as closest to canonical axis format.
            labels : optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.

        See Also: `nibabel.orientations.ornt2axcodes`.
        """
        if axcodes is None and not as_closest_canonical:
            raise ValueError('provide either `axcodes` or `as_closest_canonical=True`.')
        if axcodes is not None and as_closest_canonical:
            warnings.warn('using as_closest_canonical=True, axcodes ignored.')
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels

    def __call__(self, data_array, original_affine=None):
        """
        original orientation of `data_array` is defined by `original_affine`.

        Args:
            data_array (ndarray): in shape (num_channels, H[, W, ...]).
            original_affine (4x4 matrix): original affine.
        Returns:
            data_array (reoriented in `self.axcodes`), original axcodes, current axcodes.
        """
        sr = data_array.ndim - 1
        if sr <= 0:
            raise ValueError('the array should have at least one spatial dimension.')
        if original_affine is None:
            original_affine = np.eye(sr + 1, dtype=np.float64)
            affine = np.eye(sr + 1, dtype=np.float64)
        else:
            affine = to_affine_nd(sr, original_affine)
        src = nib.io_orientation(affine)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
            if len(dst) < sr:
                raise ValueError('`self.axcodes` should have at least {0} elements'
                                 ' given the data array is in spatial {0}D, got "{1}"'.format(sr, self.axcodes))
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        ornt = spatial_ornt.copy()
        ornt[:, 0] += 1  # skip channel dim
        ornt = np.concatenate([np.array([[0, 1]]), ornt])
        shape = data_array.shape[1:]
        data_array = nib.orientations.apply_orientation(data_array, ornt)
        new_affine = affine @ nib.orientations.inv_ornt_aff(spatial_ornt, shape)
        new_affine = to_affine_nd(original_affine, new_affine)
        return data_array, original_affine, new_affine


class LoadNifti:
    """
    Load Nifti format file or files from provided path. If loading a list of
    files, stack them together and add a new dimension as first dimension, and
    use the meta data of the first image to represent the stacked result. Note
    that the affine transform of all the images should be same if ``image_only=False``.
    """

    def __init__(self, as_closest_canonical=False, image_only=False, dtype=np.float32):
        """
        Args:
            as_closest_canonical (bool): if True, load the image as closest to canonical axis format.
            image_only (bool): if True return only the image volume, otherwise return image data array and header dict.
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.

        Note:
            The transform returns image data array if `image_only` is True,
            or a tuple of two elements containing the data array, and the Nifti
            header in a dict format otherwise.
            if a dictionary header is returned:

            - header['affine'] stores the affine of the image.
            - header['original_affine'] will be additionally created to store the original affine,
              if the current affine is different from that loaded from `filename_or_obj`.
        """
        self.as_closest_canonical = as_closest_canonical
        self.image_only = image_only
        self.dtype = dtype

    def __call__(self, filename):
        """
        Args:
            filename (str, list, tuple, file): path file or file-like object or a list of files.
        """
        filename = ensure_tuple(filename)
        img_array = list()
        compatible_meta = dict()
        for name in filename:
            img = nib.load(name)
            img = correct_nifti_header_if_necessary(img)
            header = dict(img.header)
            header['filename_or_obj'] = name
            header['affine'] = img.affine
            header['as_closest_canonical'] = self.as_closest_canonical

            if self.as_closest_canonical:
                header['original_affine'] = img.affine
                img = nib.as_closest_canonical(img)
                header['affine'] = img.affine

            img_array.append(np.array(img.get_fdata(dtype=self.dtype)))
            img.uncache()

            if self.image_only:
                continue

            if not compatible_meta:
                for meta_key in header:
                    meta_datum = header[meta_key]
                    if type(meta_datum).__name__ == 'ndarray' \
                            and np_str_obj_array_pattern.search(meta_datum.dtype.str) is not None:
                        continue
                    compatible_meta[meta_key] = meta_datum
            else:
                assert np.allclose(header['affine'], compatible_meta['affine']), \
                    'affine data of all images should be same.'

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        if self.image_only:
            return img_array
        return img_array, compatible_meta


class LoadPNG:
    """
    Load common 2D image format (PNG, JPG, etc. using PIL) file or files from provided path.
    It's based on the Image module in PIL library.
    """

    def __init__(self, dtype=np.float32):
        """Args:
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.

        """
        self.dtype = dtype

    def __call__(self, filename):
        """
        Args:
            filename (str, list, tuple, file): path file or file-like object or a list of files.
        """
        filename = ensure_tuple(filename)
        img_array = list()
        for name in filename:
            img = np.asarray(Image.open(name))
            if self.dtype:
                img = img.astype(self.dtype)
            img_array.append(img)

        return np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]


class AsChannelFirst:
    """
    Change the channel dimension of the image to the first dimension.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used to convert, for example, a channel-last image array in shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels) into the channel-first format,
    so that the multidimensional image array can be correctly interpreted by the other
    transforms.

    Args:
        channel_dim (int): which dimension of input image is the channel, default is the last dimension.
    """

    def __init__(self, channel_dim=-1):
        assert isinstance(channel_dim, int) and channel_dim >= -1, 'invalid channel dimension.'
        self.channel_dim = channel_dim

    def __call__(self, img):
        return np.moveaxis(img, self.channel_dim, 0)


class AddChannel:
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """

    def __call__(self, img):
        return img[None]


class Transpose:
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, indices):
        self.indices = indices

    def __call__(self, img):
        return img.transpose(self.indices)


class Rescale:
    """
    Rescales the input image to the given value range.
    """

    def __init__(self, minv=0.0, maxv=1.0, dtype=np.float32):
        self.minv = minv
        self.maxv = maxv
        self.dtype = dtype

    def __call__(self, img):
        return rescale_array(img, self.minv, self.maxv, self.dtype)


class GaussianNoise(Randomizable):
    """Add gaussian noise to image.

    Args:
        mean (float or array of floats): Mean or “centre” of the distribution.
        std (float): Standard deviation (spread) of distribution.
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

        self._noise = None

    def randomize(self, im_shape):
        self._noise = self.R.normal(self.mean, self.R.uniform(0, self.std), size=im_shape)

    def __call__(self, img):
        self.randomize(img.shape)
        return img + self._noise


class Flip:
    """Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        spatial_axis (None, int or tuple of ints): spatial axes along which to flip over. Default is None.
    """

    def __init__(self, spatial_axis=None):
        self.spatial_axis = spatial_axis

    def __call__(self, img):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        flipped = list()
        for channel in img:
            flipped.append(
                np.flip(channel, self.spatial_axis)
            )
        return np.stack(flipped)


class Resize:
    """
    Resize the input image to given resolution. Uses skimage.transform.resize underneath.
    For additional details, see https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize.

    Args:
        output_spatial_shape (tuple or list): expected shape of spatial dimensions after resize operation.
        order (int): Order of spline interpolation. Default=1.
        mode (str): Points outside boundaries are filled according to given mode.
            Options are 'constant', 'edge', 'symmetric', 'reflect', 'wrap'.
        cval (float): Used with mode 'constant', the value outside image boundaries.
        clip (bool): Whether to clip range of output values after interpolation. Default: True.
        preserve_range (bool): Whether to keep original range of values. Default is True.
            If False, input is converted according to conventions of img_as_float. See
            https://scikit-image.org/docs/dev/user_guide/data_types.html.
        anti_aliasing (bool): Whether to apply a gaussian filter to image before down-scaling.
            Default is True.
        anti_aliasing_sigma (float, tuple of floats): Standard deviation for gaussian filtering.
    """

    def __init__(self, output_spatial_shape, order=1, mode='reflect', cval=0,
                 clip=True, preserve_range=True, anti_aliasing=True, anti_aliasing_sigma=None):
        assert isinstance(order, int), "order must be integer."
        self.output_spatial_shape = output_spatial_shape
        self.order = order
        self.mode = mode
        self.cval = cval
        self.clip = clip
        self.preserve_range = preserve_range
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma

    def __call__(self, img):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        resized = list()
        for channel in img:
            resized.append(
                resize(channel, self.output_spatial_shape, order=self.order,
                       mode=self.mode, cval=self.cval,
                       clip=self.clip, preserve_range=self.preserve_range,
                       anti_aliasing=self.anti_aliasing,
                       anti_aliasing_sigma=self.anti_aliasing_sigma)
            )
        return np.stack(resized).astype(np.float32)


class Rotate:
    """
    Rotates an input image by given angle. Uses scipy.ndimage.rotate. For more details, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html

    Args:
        angle (float): Rotation angle in degrees.
        spatial_axes (tuple of 2 ints): Spatial axes of rotation. Default: (0, 1).
            This is the first two axis in spatial dimensions.
        reshape (bool): If reshape is true, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        order (int): Order of spline interpolation. Range 0-5. Default: 1. This is
            different from scipy where default interpolation is 3.
        mode (str): Points outside boundary filled according to this mode. Options are
            'constant', 'nearest', 'reflect', 'wrap'. Default: 'constant'.
        cval (scalar): Values to fill outside boundary. Default: 0.
        prefilter (bool): Apply spline_filter before interpolation. Default: True.
    """

    def __init__(self, angle, spatial_axes=(0, 1), reshape=True, order=1, mode='constant', cval=0, prefilter=True):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.spatial_axes = spatial_axes

    def __call__(self, img):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        rotated = list()
        for channel in img:
            rotated.append(
                scipy.ndimage.rotate(channel, self.angle, self.spatial_axes, reshape=self.reshape,
                                     order=self.order, mode=self.mode, cval=self.cval, prefilter=self.prefilter)
            )
        return np.stack(rotated).astype(np.float32)


class Zoom:
    """ Zooms a nd image. Uses scipy.ndimage.zoom or cupyx.scipy.ndimage.zoom in case of gpu.
    For details, please see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html.

    Args:
        zoom (float or sequence): The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        order (int): order of interpolation. Default=3.
        mode (str): Determines how input is extended beyond boundaries. Default is 'constant'.
        cval (scalar, optional): Value to fill past edges. Default is 0.
        use_gpu (bool): Should use cpu or gpu. Uses cupyx which doesn't support order > 1 and modes
            'wrap' and 'reflect'. Defaults to cpu for these cases or if cupyx not found.
        keep_size (bool): Should keep original size (pad if needed).
    """

    def __init__(self, zoom, order=3, mode='constant', cval=0, prefilter=True, use_gpu=False, keep_size=False):
        assert isinstance(order, int), "Order must be integer."
        self.zoom = zoom
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.use_gpu = use_gpu
        self.keep_size = keep_size

        if self.use_gpu:
            try:
                from cupyx.scipy.ndimage import zoom as zoom_gpu

                self._zoom = zoom_gpu
            except ImportError:
                print('For GPU zoom, please install cupy. Defaulting to cpu.')
                self._zoom = scipy.ndimage.zoom
                self.use_gpu = False
        else:
            self._zoom = scipy.ndimage.zoom

    def __call__(self, img):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        zoomed = list()
        if self.use_gpu:
            import cupy
            for channel in cupy.array(img):
                zoom_channel = self._zoom(channel,
                                          zoom=self.zoom,
                                          order=self.order,
                                          mode=self.mode,
                                          cval=self.cval,
                                          prefilter=self.prefilter)
                zoomed.append(cupy.asnumpy(zoom_channel))
        else:
            for channel in img:
                zoomed.append(
                    self._zoom(channel,
                               zoom=self.zoom,
                               order=self.order,
                               mode=self.mode,
                               cval=self.cval,
                               prefilter=self.prefilter))
        zoomed = np.stack(zoomed).astype(np.float32)

        if not self.keep_size or np.allclose(img.shape, zoomed.shape):
            return zoomed

        pad_vec = [[0, 0]] * len(img.shape)
        slice_vec = [slice(None)] * len(img.shape)
        for idx, (od, zd) in enumerate(zip(img.shape, zoomed.shape)):
            diff = od - zd
            half = abs(diff) // 2
            if diff > 0:  # need padding
                pad_vec[idx] = [half, diff - half]
            elif diff < 0:  # need slicing
                slice_vec[idx] = slice(half, half + od)
        zoomed = np.pad(zoomed, pad_vec, mode='constant')
        return zoomed[tuple(slice_vec)]


class ToTensor:
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img):
        if torch.is_tensor(img):
            return img.contiguous()
        return torch.as_tensor(np.ascontiguousarray(img))


class RandUniformPatch(Randomizable):
    """
    Selects a patch of the given size chosen at a uniformly random position in the image.

    Args:
        patch_spatial_size (tuple or list): Expected patch size of spatial dimensions.
    """

    def __init__(self, patch_spatial_size):
        self.patch_spatial_size = (None,) + tuple(patch_spatial_size)

        self._slices = None

    def randomize(self, image_shape, patch_shape):
        self._slices = get_random_patch(image_shape, patch_shape, self.R)

    def __call__(self, img):
        patch_spatial_size = get_valid_patch_size(img.shape, self.patch_spatial_size)
        self.randomize(img.shape, patch_spatial_size)
        return img[self._slices]


class NormalizeIntensity:
    """Normalize input based on provided args, using calculated mean and std if not provided
    (shape of subtrahend and divisor must match. if 0, entire volume uses same subtrahend and
    divisor, otherwise the shape can have dimension 1 for channels).

    Args:
        subtrahend (ndarray): the amount to subtract by (usually the mean)
        divisor (ndarray): the amount to divide by (usually the standard deviation)
    """

    def __init__(self, subtrahend=None, divisor=None):
        if subtrahend is not None or divisor is not None:
            assert isinstance(subtrahend, np.ndarray) and isinstance(divisor, np.ndarray), \
                'subtrahend and divisor must be set in pair and in numpy array.'
        self.subtrahend = subtrahend
        self.divisor = divisor

    def __call__(self, img):
        if self.subtrahend is not None and self.divisor is not None:
            img -= self.subtrahend
            img /= self.divisor
        else:
            img -= np.mean(img)
            img /= np.std(img)

        return img


class ScaleIntensityRange:
    """Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    Args:
        a_min (int or float): intensity original range min.
        a_max (int or float): intensity original range max.
        b_min (int or float): intensity target range min.
        b_max (int or float): intensity target range max.
        clip (bool): whether to perform clip after scaling.
    """

    def __init__(self, a_min, a_max, b_min, b_max, clip=False):
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def __call__(self, img):
        img = (img - self.a_min) / (self.a_max - self.a_min)
        img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)

        return img


class PadImageEnd:
    """Performs padding by appending to the end of the data all on one side for each dimension.
     Uses np.pad so in practice, a mode needs to be provided. See numpy.lib.arraypad.pad
     for additional details.

    Args:
        out_size (list): the size of region of interest at the end of the operation.
        mode (string): a portion from numpy.lib.arraypad.pad is copied below.
    """

    def __init__(self, out_size, mode):
        assert isinstance(out_size, (list, tuple)), 'out_size must be list or tuple.'
        self.out_size = out_size
        assert isinstance(mode, str), 'mode must be str.'
        self.mode = mode

    def _determine_data_pad_width(self, data_shape):
        return [(0, max(self.out_size[i] - data_shape[i], 0)) for i in range(len(self.out_size))]

    def __call__(self, img):
        data_pad_width = self._determine_data_pad_width(img.shape[2:])
        all_pad_width = [(0, 0), (0, 0)] + data_pad_width
        img = np.pad(img, all_pad_width, self.mode)
        return img


class Rotate90:
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    """

    def __init__(self, k=1, spatial_axes=(0, 1)):
        """
        Args:
            k (int): number of times to rotate by 90 degrees.
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.k = k
        self.spatial_axes = spatial_axes

    def __call__(self, img):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        rotated = list()
        for channel in img:
            rotated.append(
                np.rot90(channel, self.k, self.spatial_axes)
            )
        return np.stack(rotated)


class RandRotate90(Randomizable):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(self, prob=0.1, max_k=3, spatial_axes=(0, 1)):
        """
        Args:
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self):
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate90(self._rand_k, self.spatial_axes)
        return rotator(img)


class SpatialCrop:
    """General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively if center and size
    are not provided, the start and end coordinates of the ROI must be provided.
    The sub-volume must sit the within original image.

    Note: This transform will not work if the crop region is larger than the image itself.
    """

    def __init__(self, roi_center=None, roi_size=None, roi_start=None, roi_end=None):
        """
        Args:
            roi_center (list or tuple): voxel coordinates for center of the crop ROI.
            roi_size (list or tuple): size of the crop ROI.
            roi_start (list or tuple): voxel coordinates for start of the crop ROI.
            roi_end (list or tuple): voxel coordinates for end of the crop ROI.
        """
        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.uint16)
            roi_size = np.asarray(roi_size, dtype=np.uint16)
            self.roi_start = np.subtract(roi_center, np.floor_divide(roi_size, 2))
            self.roi_end = np.add(self.roi_start, roi_size)
        else:
            assert roi_start is not None and roi_end is not None, 'roi_start and roi_end must be provided.'
            self.roi_start = np.asarray(roi_start, dtype=np.uint16)
            self.roi_end = np.asarray(roi_end, dtype=np.uint16)

        assert np.all(self.roi_start >= 0), 'all elements of roi_start must be greater than or equal to 0.'
        assert np.all(self.roi_end > 0), 'all elements of roi_end must be positive.'
        assert np.all(self.roi_end >= self.roi_start), 'invalid roi range.'

    def __call__(self, img):
        max_end = img.shape[1:]
        sd = min(len(self.roi_start), len(max_end))
        assert np.all(max_end[:sd] >= self.roi_start[:sd]), 'roi start out of image space.'
        assert np.all(max_end[:sd] >= self.roi_end[:sd]), 'roi end out of image space.'

        slices = [slice(None)] + [slice(s, e) for s, e in zip(self.roi_start[:sd], self.roi_end[:sd])]
        data = img[tuple(slices)].copy()
        return data


class RandRotate(Randomizable):
    """Randomly rotates the input arrays.

    Args:
        prob (float): Probability of rotation.
        degrees (tuple of float or float): Range of rotation in degrees. If single number,
            angle is picked from (-degrees, degrees).
        spatial_axes (tuple of 2 ints): Spatial axes of rotation. Default: (0, 1).
            This is the first two axis in spatial dimensions.
        reshape (bool): If reshape is true, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        order (int): Order of spline interpolation. Range 0-5. Default: 1. This is
            different from scipy where default interpolation is 3.
        mode (str): Points outside boundary filled according to this mode. Options are
            'constant', 'nearest', 'reflect', 'wrap'. Default: 'constant'.
        cval (scalar): Value to fill outside boundary. Default: 0.
        prefilter (bool): Apply spline_filter before interpolation. Default: True.
    """

    def __init__(self, degrees, prob=0.1, spatial_axes=(0, 1), reshape=True, order=1,
                 mode='constant', cval=0, prefilter=True):
        self.prob = prob
        self.degrees = degrees
        self.reshape = reshape
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.spatial_axes = spatial_axes

        if not hasattr(self.degrees, '__iter__'):
            self.degrees = (-self.degrees, self.degrees)
        assert len(self.degrees) == 2, "degrees should be a number or pair of numbers."

        self._do_transform = False
        self.angle = None

    def randomize(self):
        self._do_transform = self.R.random_sample() < self.prob
        self.angle = self.R.uniform(low=self.degrees[0], high=self.degrees[1])

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate(self.angle, self.spatial_axes, self.reshape, self.order,
                         self.mode, self.cval, self.prefilter)
        return rotator(img)


class RandFlip(Randomizable):
    """Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob (float): Probability of flipping.
        spatial_axis (None, int or tuple of ints): Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, prob=0.1, spatial_axis=None):
        self.prob = prob
        self.flipper = Flip(spatial_axis=spatial_axis)
        self._do_transform = False

    def randomize(self):
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        return self.flipper(img)


class RandZoom(Randomizable):
    """Randomly zooms input arrays with given probability within given zoom range.

    Args:
        prob (float): Probability of zooming.
        min_zoom (float or sequence): Min zoom factor. Can be float or sequence same size as image.
            If a float, min_zoom is the same for each spatial axis.
            If a sequence, min_zoom should contain one value for each spatial axis.
        max_zoom (float or sequence): Max zoom factor. Can be float or sequence same size as image.
            If a float, max_zoom is the same for each spatial axis.
            If a sequence, max_zoom should contain one value for each spatial axis.
        order (int): order of interpolation. Default=3.
        mode ('reflect', 'constant', 'nearest', 'mirror', 'wrap'): Determines how input is
            extended beyond boundaries. Default: 'constant'.
        cval (scalar, optional): Value to fill past edges. Default is 0.
        use_gpu (bool): Should use cpu or gpu. Uses cupyx which doesn't support order > 1 and modes
            'wrap' and 'reflect'. Defaults to cpu for these cases or if cupyx not found.
        keep_size (bool): Should keep original size (pad if needed).
    """

    def __init__(self, prob=0.1, min_zoom=0.9, max_zoom=1.1, order=3,
                 mode='constant', cval=0, prefilter=True,
                 use_gpu=False, keep_size=False):
        if hasattr(min_zoom, '__iter__') and hasattr(max_zoom, '__iter__'):
            assert len(min_zoom) == len(max_zoom), "min_zoom and max_zoom must have same length."
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.prob = prob
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.use_gpu = use_gpu
        self.keep_size = keep_size

        self._do_transform = False
        self._zoom = None

    def randomize(self):
        self._do_transform = self.R.random_sample() < self.prob
        if hasattr(self.min_zoom, '__iter__'):
            self._zoom = (self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom))
        else:
            self._zoom = self.R.uniform(self.min_zoom, self.max_zoom)

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        zoomer = Zoom(self._zoom, self.order, self.mode, self.cval, self.prefilter, self.use_gpu, self.keep_size)
        return zoomer(img)


class AffineGrid:
    """
    Affine transforms on the coordinates.
    """

    def __init__(self,
                 rotate_params=None,
                 shear_params=None,
                 translate_params=None,
                 scale_params=None,
                 as_tensor_output=True,
                 device=None):
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params

        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(self, spatial_size=None, grid=None):
        """
        Args:
            spatial_size (list or tuple of int): output grid size.
            grid (ndarray): grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError('Either specify a grid or a spatial size to create a grid from.')

        spatial_dims = len(grid.shape) - 1
        affine = np.eye(spatial_dims + 1)
        if self.rotate_params:
            affine = affine @ create_rotate(spatial_dims, self.rotate_params)
        if self.shear_params:
            affine = affine @ create_shear(spatial_dims, self.shear_params)
        if self.translate_params:
            affine = affine @ create_translate(spatial_dims, self.translate_params)
        if self.scale_params:
            affine = affine @ create_scale(spatial_dims, self.scale_params)
        affine = torch.as_tensor(np.ascontiguousarray(affine), device=self.device)

        grid = torch.as_tensor(np.ascontiguousarray(grid)) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            grid = grid.to(self.device)
        grid = (affine.float() @ grid.reshape((grid.shape[0], -1)).float()).reshape([-1] + list(grid.shape[1:]))
        if self.as_tensor_output:
            return grid
        return grid.cpu().numpy()


class RandAffineGrid(Randomizable):
    """
    generate randomised affine grid
    """

    def __init__(self,
                 rotate_range=None,
                 shear_range=None,
                 translate_range=None,
                 scale_range=None,
                 as_tensor_output=True,
                 device=None):
        """
        Args:
            rotate_range (a sequence of positive floats): rotate_range[0] with be used to generate the 1st rotation
                parameter from `uniform[-rotate_range[0], rotate_range[0])`. Similarly, `rotate_range[2]` and
                `rotate_range[3]` are used in 3D affine for the range of 2nd and 3rd axes.
            shear_range (a sequence of positive floats): shear_range[0] with be used to generate the 1st shearing
                parameter from `uniform[-shear_range[0], shear_range[0])`. Similarly, `shear_range[1]` to
                `shear_range[N]` controls the range of the uniform distribution used to generate the 2nd to
                N-th parameter.
            translate_range (a sequence of positive floats): translate_range[0] with be used to generate the 1st
                shift parameter from `uniform[-translate_range[0], translate_range[0])`. Similarly, `translate_range[1]`
                to `translate_range[N]` controls the range of the uniform distribution used to generate
                the 2nd to N-th parameter.
            scale_range (a sequence of positive floats): scaling_range[0] with be used to generate the 1st scaling
                factor from `uniform[-scale_range[0], scale_range[0]) + 1.0`. Similarly, `scale_range[1]` to
                `scale_range[N]` controls the range of the uniform distribution used to generate the 2nd to
                N-th parameter.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`
        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params = None
        self.shear_params = None
        self.translate_params = None
        self.scale_params = None

        self.as_tensor_output = as_tensor_output
        self.device = device

    def randomize(self):
        if self.rotate_range:
            self.rotate_params = [self.R.uniform(-f, f) for f in self.rotate_range if f is not None]
        if self.shear_range:
            self.shear_params = [self.R.uniform(-f, f) for f in self.shear_range if f is not None]
        if self.translate_range:
            self.translate_params = [self.R.uniform(-f, f) for f in self.translate_range if f is not None]
        if self.scale_range:
            self.scale_params = [self.R.uniform(-f, f) + 1.0 for f in self.scale_range if f is not None]

    def __call__(self, spatial_size=None, grid=None):
        """
        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        self.randomize()
        affine_grid = AffineGrid(rotate_params=self.rotate_params, shear_params=self.shear_params,
                                 translate_params=self.translate_params, scale_params=self.scale_params,
                                 as_tensor_output=self.as_tensor_output, device=self.device)
        return affine_grid(spatial_size, grid)


class RandDeformGrid(Randomizable):
    """
    generate random deformation grid
    """

    def __init__(self, spacing, magnitude_range, as_tensor_output=True, device=None):
        """
        Args:
            spacing (2 or 3 ints): spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range (2 ints): the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            as_tensor_output (bool): whether to output tensor instead of numpy array.
                defaults to True.
            device (torch device): device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range

        self.rand_mag = 1.0
        self.as_tensor_output = as_tensor_output
        self.random_offset = 0.0
        self.device = device

    def randomize(self, grid_size):
        self.random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size)))
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size):
        control_grid = create_control_grid(spatial_size, self.spacing)
        self.randomize(control_grid.shape[1:])
        control_grid[:len(spatial_size)] += self.rand_mag * self.random_offset
        if self.as_tensor_output:
            control_grid = torch.as_tensor(np.ascontiguousarray(control_grid), device=self.device)
        return control_grid


class Resample:

    def __init__(self, padding_mode='zeros', as_tensor_output=False, device=None):
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices. Defaults to 'zeros'.
            as_tensor_output(bool): whether to return a torch tensor. Defaults to False.
            device (torch.device): device on which the tensor will be allocated.
        """
        self.padding_mode = padding_mode
        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(self, img, grid, mode='bilinear'):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W[, D]).
            grid (ndarray or tensor): shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode ('nearest'|'bilinear'): interpolation order. Defaults to 'bilinear'.
        """
        if not torch.is_tensor(img):
            img = torch.as_tensor(np.ascontiguousarray(img))
        grid = torch.as_tensor(np.ascontiguousarray(grid)) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            img = img.to(self.device)
            grid = grid.to(self.device)

        for i, dim in enumerate(img.shape[1:]):
            grid[i] = 2. * grid[i] / (dim - 1.)
        grid = grid[:-1] / grid[-1:]
        grid = grid[range(img.ndim - 2, -1, -1)]
        grid = grid.permute(list(range(grid.ndim))[1:] + [0])
        out = torch.nn.functional.grid_sample(img[None].float(),
                                              grid[None].float(),
                                              mode=mode,
                                              padding_mode=self.padding_mode,
                                              align_corners=False)[0]
        if self.as_tensor_output:
            return out
        return out.cpu().numpy()


class Affine:
    """
    transform ``img`` given the affine parameters.
    """

    def __init__(self,
                 rotate_params=None,
                 shear_params=None,
                 translate_params=None,
                 scale_params=None,
                 spatial_size=None,
                 mode='bilinear',
                 padding_mode='zeros',
                 as_tensor_output=False,
                 device=None):
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            rotate_params (float, list of floats): a rotation angle in radians,
                a scalar for 2D image, a tuple of 3 floats for 3D. Defaults to no rotation.
            shear_params (list of floats):
                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params (list of floats):
                a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in pixel/voxel
                relative to the center of the input image. Defaults to no translation.
            scale_params (list of floats):
                a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Defaults to no scaling.
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to 'bilinear'.
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices. Defaults to 'zeros'.
            as_tensor_output (bool): the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.
        """
        self.affine_grid = AffineGrid(rotate_params=rotate_params,
                                      shear_params=shear_params,
                                      translate_params=translate_params,
                                      scale_params=scale_params,
                                      as_tensor_output=True,
                                      device=device)
        self.resampler = Resample(padding_mode=padding_mode, as_tensor_output=as_tensor_output, device=device)
        self.spatial_size = spatial_size
        self.mode = mode

    def __call__(self, img, spatial_size=None, mode=None):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W[, D]),
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to 'bilinear'.
        """
        spatial_size = spatial_size or self.spatial_size
        mode = mode or self.mode
        grid = self.affine_grid(spatial_size=spatial_size)
        return self.resampler(img=img, grid=grid, mode=mode)


class RandAffine(Randomizable):
    """
    Random affine transform.
    """

    def __init__(self,
                 prob=0.1,
                 rotate_range=None,
                 shear_range=None,
                 translate_range=None,
                 scale_range=None,
                 spatial_size=None,
                 mode='bilinear',
                 padding_mode='zeros',
                 as_tensor_output=True,
                 device=None):
        """
        Args:
            prob (float): probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to 'bilinear'.
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices. Defaults to 'zeros'.
            as_tensor_output (bool): the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """

        self.rand_affine_grid = RandAffineGrid(rotate_range=rotate_range, shear_range=shear_range,
                                               translate_range=translate_range, scale_range=scale_range,
                                               as_tensor_output=True, device=device)
        self.resampler = Resample(padding_mode=padding_mode, as_tensor_output=as_tensor_output, device=device)

        self.spatial_size = spatial_size
        self.mode = mode

        self.do_transform = False
        self.prob = prob

    def set_random_state(self, seed=None, state=None):
        self.rand_affine_grid.set_random_state(seed, state)
        Randomizable.set_random_state(self, seed, state)
        return self

    def randomize(self):
        self.do_transform = self.R.rand() < self.prob
        self.rand_affine_grid.randomize()

    def __call__(self, img, spatial_size=None, mode=None):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W[, D]),
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to 'bilinear'.
        """
        self.randomize()
        spatial_size = spatial_size or self.spatial_size
        mode = mode or self.mode
        if self.do_transform:
            grid = self.rand_affine_grid(spatial_size=spatial_size)
        else:
            grid = create_grid(spatial_size)
        return self.resampler(img=img, grid=grid, mode=mode)


class Rand2DElastic(Randomizable):
    """
    Random elastic deformation and affine in 2D
    """

    def __init__(self,
                 spacing,
                 magnitude_range,
                 prob=0.1,
                 rotate_range=None,
                 shear_range=None,
                 translate_range=None,
                 scale_range=None,
                 spatial_size=None,
                 mode='bilinear',
                 padding_mode='zeros',
                 as_tensor_output=False,
                 device=None):
        """
        Args:
            spacing (2 ints): distance in between the control points.
            magnitude_range (2 ints): the random offsets will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob (float): probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            spatial_size (2 ints): specifying output image spatial size [h, w].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to ``'bilinear'``.
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices.
                Defaults to ``'zeros'``.
            as_tensor_output (bool): the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        self.deform_grid = RandDeformGrid(spacing=spacing, magnitude_range=magnitude_range,
                                          as_tensor_output=True, device=device)
        self.rand_affine_grid = RandAffineGrid(rotate_range=rotate_range, shear_range=shear_range,
                                               translate_range=translate_range, scale_range=scale_range,
                                               as_tensor_output=True, device=device)
        self.resampler = Resample(padding_mode=padding_mode, as_tensor_output=as_tensor_output, device=device)

        self.spatial_size = spatial_size
        self.mode = mode
        self.prob = prob
        self.do_transform = False

    def set_random_state(self, seed=None, state=None):
        self.deform_grid.set_random_state(seed, state)
        self.rand_affine_grid.set_random_state(seed, state)
        Randomizable.set_random_state(self, seed, state)
        return self

    def randomize(self, spatial_size):
        self.do_transform = self.R.rand() < self.prob
        self.deform_grid.randomize(spatial_size)
        self.rand_affine_grid.randomize()

    def __call__(self, img, spatial_size=None, mode=None):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W),
            spatial_size (2 ints): specifying output image spatial size [h, w].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to ``self.mode``.
        """
        spatial_size = spatial_size or self.spatial_size
        self.randomize(spatial_size)
        mode = mode or self.mode
        if self.do_transform:
            grid = self.deform_grid(spatial_size=spatial_size)
            grid = self.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(grid[None], spatial_size, mode='bicubic', align_corners=False)[0]
        else:
            grid = create_grid(spatial_size)
        return self.resampler(img, grid, mode)


class Rand3DElastic(Randomizable):
    """
    Random elastic deformation and affine in 3D
    """

    def __init__(self,
                 sigma_range,
                 magnitude_range,
                 prob=0.1,
                 rotate_range=None,
                 shear_range=None,
                 translate_range=None,
                 scale_range=None,
                 spatial_size=None,
                 mode='bilinear',
                 padding_mode='zeros',
                 as_tensor_output=False,
                 device=None):
        """
        Args:
            sigma_range (2 ints): a Gaussian kernel with standard deviation sampled
                 from ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range (2 ints): the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob (float): probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            spatial_size (3 ints): specifying output image spatial size [h, w, d].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to ``'bilinear'``.
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices.
                Defaults to ``'zeros'``.
            as_tensor_output (bool): the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        self.rand_affine_grid = RandAffineGrid(rotate_range, shear_range, translate_range, scale_range, True, device)
        self.resampler = Resample(padding_mode=padding_mode, as_tensor_output=as_tensor_output, device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = mode
        self.device = device

        self.prob = prob
        self.do_transform = False
        self.rand_offset = None
        self.magnitude = 1.0
        self.sigma = 1.0

    def set_random_state(self, seed=None, state=None):
        self.rand_affine_grid.set_random_state(seed, state)
        Randomizable.set_random_state(self, seed, state)
        return self

    def randomize(self, grid_size):
        self.do_transform = self.R.rand() < self.prob
        if self.do_transform:
            self.rand_offset = self.R.uniform(-1., 1., [3] + list(grid_size))
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(self, img, spatial_size=None, mode=None):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W, D),
            spatial_size (3 ints): specifying spatial 3D output image spatial size [h, w, d].
            mode ('nearest'|'bilinear'): interpolation order. Defaults to 'self.mode'.
        """
        spatial_size = spatial_size or self.spatial_size
        mode = mode or self.mode
        self.randomize(spatial_size)
        grid = create_grid(spatial_size)
        if self.do_transform:
            grid = torch.as_tensor(np.ascontiguousarray(grid), device=self.device)
            gaussian = GaussianFilter(3, self.sigma, 3., device=self.device)
            grid[:3] += gaussian(self.rand_offset[None])[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        return self.resampler(img, grid, mode)
