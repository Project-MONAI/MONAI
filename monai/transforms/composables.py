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
A collection of dictionary-based wrappers around the "vanilla" transforms
defined in :py:class:`monai.transforms.transforms`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import numpy as np
import torch

from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.transforms import (AddChannel, AsChannelFirst, Flip, LoadNifti, NormalizeIntensity, Orientation,
                                         Rand2DElastic, Rand3DElastic, RandAffine, Rescale, Resize, Rotate, Rotate90,
                                         ScaleIntensityRange, Spacing, SpatialCrop, Zoom, ToTensor, LoadPNG)
from monai.transforms.utils import (create_grid, generate_pos_neg_label_crop_centers)
from monai.utils.misc import ensure_tuple


class Spacingd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transforms.Spacing`.

    This transform assumes the ``data`` dictionary has a field for the input
    data's affine.  The field is created by either ``meta_key_format.format(key,
    'affine')`` or ``meta_key_format.format(key, 'original_affine')``.

    After resampling the input array, this transform will write the affine
    after resampling to the field ``meta_key_format.format(key, 'affine')``,
    at the same time, if ``meta_key_format.format(key, 'original_affine')`` doesn't exist,
    the field will be created and set to the affine before resampling.

    if no affine is specified in the input data, defauting to "eye(4)".

    see also:
        :py:class:`monai.transforms.transforms.Spacing`
    """

    def __init__(self, keys, pixdim, diagonal=False, mode='constant', cval=0,
                 interp_order=3, dtype=None, meta_key_format='{}.{}'):
        """
        Args:
            pixdim (sequence of floats): output voxel spacing.
            diagonal (bool): whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, pixdim_2, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, the axes orientation, orthogonal rotation and
                translations components from the original affine will be
                preserved in the target affine. This option will not flip/swap
                axes against the original ones.
            mode (`reflect|constant|nearest|mirror|wrap`):
                The mode parameter determines how the input array is extended beyond its boundaries.
                Default is 'constant'.
            cval (scalar): Value to fill past edges of input if mode is "constant". Default is 0.0.
            interp_order (int or sequence of ints): int: the same interpolation order
                for all data indexed by `self.keys`; sequence of ints, should
                correspond to an interpolation order for each data item indexed
                by `self.keys` respectively.
            dtype (None or np.dtype): output array data type, defaults to None to use input data's dtype.
            meta_key_format (str): key format to read/write affine matrices to the data dictionary.
        """
        MapTransform.__init__(self, keys)
        self.spacing_transform = Spacing(pixdim, diagonal=diagonal, mode=mode, cval=cval, dtype=dtype)
        interp_order = ensure_tuple(interp_order)
        self.interp_order = interp_order \
            if len(interp_order) == len(self.keys) else interp_order * len(self.keys)
        self.meta_key_format = meta_key_format

    def __call__(self, data):
        d = dict(data)
        for key, interp in zip(self.keys, self.interp_order):
            affine_key = self.meta_key_format.format(key, 'affine')
            original_key = self.meta_key_format.format(key, 'original_affine')
            affine = d.get(affine_key, None)
            if affine is None:
                affine = d.get(original_key, None)
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            d[key], affine_, new_affine = self.spacing_transform(
                data_array=d[key], original_affine=affine, interp_order=interp)
            if d.get(original_key, None) is None:
                # set the 'original_affine' field
                d[original_key] = affine_
            # set the 'affine' key
            d[affine_key] = new_affine
        return d


class Orientationd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transforms.Orientation`.

    This transform assumes the ``data`` dictionary has a field for the input
    data's affine.  The field is created by either ``meta_key_format.format(key,
    'affine')`` or ``meta_key_format.format(key, 'original_affine')``.

    After reorientate the input array, this transform will store the current
    affine in the ``data`` dictionary,
    at the same time, if ``meta_key_format.format(key, 'original_affine')`` doesn't exist,
    the field will be created and set to the affine before resampling.
    """

    def __init__(self, keys, axcodes=None, as_closest_canonical=False,
                 labels=tuple(zip('LPI', 'RAS')), meta_key_format='{}.{}'):
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
            meta_key_format (str): key format to read/write affine matrices to the data dictionary.

        See Also:
            `nibabel.orientations.ornt2axcodes`.
        """
        MapTransform.__init__(self, keys)
        self.ornt_transform = Orientation(
            axcodes=axcodes, as_closest_canonical=as_closest_canonical, labels=labels)
        self.meta_key_format = meta_key_format

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            affine_key = self.meta_key_format.format(key, 'affine')
            original_key = self.meta_key_format.format(key, 'original_affine')

            affine = d.get(affine_key, None)
            if affine is None:
                affine = d.get(original_key, None)
            d[key], affine_, new_affine = self.ornt_transform(d[key], affine)
            if d.get(original_key, None) is None:
                d[original_key] = affine_
            d[affine_key] = new_affine
        return d


class LoadNiftid(MapTransform):
    """
    Dictionary-based wrapper of LoadNifti, must load image and metadata
    together. If loading a list of files in one key, stack them together and
    add a new dimension as the first dimension, and use the meta data of the
    first image to represent the stacked result. Note that the affine transform
    of all the stacked images should be same. The output metadata field will be created as
    ``self.meta_key_format(key, metadata_key)``.
    """

    def __init__(self, keys, as_closest_canonical=False, dtype=np.float32,
                 meta_key_format='{}.{}', overwriting_keys=False):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            as_closest_canonical (bool): if True, load the image as closest to canonical axis format.
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.
            meta_key_format (str): key format to store meta data of the nifti image.
                it must contain 2 fields for the key of this image and the key of every meta data item.
            overwriting_keys (bool): whether allow to overwrite existing keys of meta data.
                default is False, which will raise exception if encountering existing key.
        """
        MapTransform.__init__(self, keys)
        self.loader = LoadNifti(as_closest_canonical, False, dtype)
        self.meta_key_format = meta_key_format
        self.overwriting_keys = overwriting_keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            data = self.loader(d[key])
            assert isinstance(data, (tuple, list)), 'loader must return a tuple or list.'
            d[key] = data[0]
            assert isinstance(data[1], dict), 'metadata must be a dict.'
            for k in sorted(data[1]):
                key_to_add = self.meta_key_format.format(key, k)
                if key_to_add in d and not self.overwriting_keys:
                    raise KeyError('meta data key {} already exists.'.format(key_to_add))
                d[key_to_add] = data[1][k]
        return d


class LoadPNGd(MapTransform):
    """
    dictionary-based wrapper of LoadPNG.
    """

    def __init__(self, keys, dtype=np.float32):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.
        """
        MapTransform.__init__(self, keys)
        self.loader = LoadPNG(dtype)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.loader(d[key])
        return d


class AsChannelFirstd(MapTransform):
    """
    dictionary-based wrapper of AsChannelFirst.
    """

    def __init__(self, keys, channel_dim=-1):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim (int): which dimension of input image is the channel, default is the last dimension.
        """
        MapTransform.__init__(self, keys)
        self.converter = AsChannelFirst(channel_dim=channel_dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AddChanneld(MapTransform):
    """
    dictionary-based wrapper of AddChannel.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        MapTransform.__init__(self, keys)
        self.adder = AddChannel()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.adder(d[key])
        return d


class ToTensord(MapTransform):
    """
    dictionary-based wrapper of ToTensor.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        MapTransform.__init__(self, keys)
        self.converter = ToTensor()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class Rotate90d(MapTransform):
    """
    dictionary-based wrapper of Rotate90.
    """

    def __init__(self, keys, k=1, spatial_axes=(0, 1)):
        """
        Args:
            k (int): number of times to rotate by 90 degrees.
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        MapTransform.__init__(self, keys)
        self.k = k
        self.spatial_axes = spatial_axes

        self.rotator = Rotate90(self.k, self.spatial_axes)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.rotator(d[key])
        return d


class Rescaled(MapTransform):
    """
    dictionary-based wrapper of Rescale.
    """

    def __init__(self, keys, minv=0.0, maxv=1.0, dtype=np.float32):
        MapTransform.__init__(self, keys)
        self.rescaler = Rescale(minv, maxv, dtype)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.rescaler(d[key])
        return d


class Resized(MapTransform):
    """
    dictionary-based wrapper of Resize.

    Args:
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        output_spatial_shape (tuple or list): expected shape of spatial dimensions after resize operation.
        order (int): Order of spline interpolation. Default=1.
        mode (str): Points outside boundaries are filled according to given mode.
            Options are 'constant', 'edge', 'symmetric', 'reflect', 'wrap'.
        cval (float): Used with mode 'constant', the value outside image boundaries.
        clip (bool): Whether to clip range of output values after interpolation. Default: True.
        preserve_range (bool): Whether to keep original range of values. Default is True.
            If False, input is converted according to conventions of img_as_float. See
            https://scikit-image.org/docs/dev/user_guide/data_types.html.
        anti_aliasing (bool): Whether to apply a gaussian filter to image before down-scaling. Default is True.
        anti_aliasing_sigma (float, tuple of floats): Standard deviation for gaussian filtering.
    """

    def __init__(self, keys, output_spatial_shape, order=1, mode='reflect', cval=0,
                 clip=True, preserve_range=True, anti_aliasing=True, anti_aliasing_sigma=None):
        MapTransform.__init__(self, keys)
        self.resizer = Resize(output_spatial_shape, order, mode, cval, clip, preserve_range,
                              anti_aliasing, anti_aliasing_sigma)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.resizer(d[key])
        return d


class RandUniformPatchd(Randomizable, MapTransform):
    """
    Selects a patch of the given size chosen at a uniformly random position in the image.

    Args:
        patch_spatial_size (tuple or list): Expected patch size of spatial dimensions.
    """

    def __init__(self, keys, patch_spatial_size):
        MapTransform.__init__(self, keys)

        self.patch_spatial_size = (None,) + tuple(patch_spatial_size)

        self._slices = None

    def randomize(self, image_shape, patch_shape):
        self._slices = get_random_patch(image_shape, patch_shape, self.R)

    def __call__(self, data):
        d = dict(data)

        image_shape = d[self.keys[0]].shape  # image shape from the first data key
        patch_spatial_size = get_valid_patch_size(image_shape, self.patch_spatial_size)
        self.randomize(image_shape, patch_spatial_size)
        for key in self.keys:
            d[key] = d[key][self._slices]
        return d


class RandRotate90d(Randomizable, MapTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(self, keys, prob=0.1, max_k=3, spatial_axes=(0, 1)):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        MapTransform.__init__(self, keys)

        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self):
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        self.randomize()
        if not self._do_transform:
            return data

        rotator = Rotate90(self._rand_k, self.spatial_axes)
        d = dict(data)
        for key in self.keys:
            d[key] = rotator(d[key])
        return d


class NormalizeIntensityd(MapTransform):
    """
    dictionary-based wrapper of NormalizeIntensity.

    Args:
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transform.composables.MapTransform
        subtrahend (ndarray): the amount to subtract by (usually the mean)
        divisor (ndarray): the amount to divide by (usually the standard deviation)
    """

    def __init__(self, keys, subtrahend=None, divisor=None):
        MapTransform.__init__(self, keys)
        self.normalizer = NormalizeIntensity(subtrahend, divisor)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d


class ScaleIntensityRanged(MapTransform):
    """
    dictionary-based wrapper of ScaleIntensityRange.

    Args:
        keys (hashable items): keys of the corresponding items to be transformed.
            See also: monai.transform.composables.MapTransform
        a_min (int or float): intensity original range min.
        a_max (int or float): intensity original range max.
        b_min (int or float): intensity target range min.
        b_max (int or float): intensity target range max.
        clip (bool): whether to perform clip after scaling.
    """

    def __init__(self, keys, a_min, a_max, b_min, b_max, clip=False):
        MapTransform.__init__(self, keys)
        self.scaler = ScaleIntensityRange(a_min, a_max, b_min, b_max, clip)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.scaler(d[key])
        return d


class RandCropByPosNegLabeld(Randomizable, MapTransform):
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys (list): parameter will be used to get and set the actual data item to transform.
        label_key (str): name of key for label image, this will be used for finding foreground/background.
        size (list, tuple): the size of the crop region e.g. [224,224,128]
        pos (int, float): used to calculate the ratio ``pos / (pos + neg)`` for the probability to pick a
          foreground voxel as a center rather than a background voxel.
        neg (int, float): used to calculate the ratio ``pos / (pos + neg)`` for the probability to pick a
          foreground voxel as a center rather than a background voxel.
        num_samples (int): number of samples (crop regions) to take in each list.
        image_key (str): if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold (int or float): if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
    """

    def __init__(self, keys, label_key, size, pos=1, neg=1, num_samples=1, image_key=None, image_threshold=0):
        MapTransform.__init__(self, keys)
        assert isinstance(label_key, str), 'label_key must be a string.'
        assert isinstance(size, (list, tuple)), 'size must be list or tuple.'
        assert all(isinstance(x, int) and x > 0 for x in size), 'all elements of size must be positive integers.'
        assert float(pos) >= 0 and float(neg) >= 0, "pos and neg must be greater than or equal to 0."
        assert float(pos) + float(neg) > 0, "pos and neg cannot both be 0."
        assert isinstance(num_samples, int), \
            "invalid samples number: {}. num_samples must be an integer.".format(num_samples)
        assert num_samples >= 0, 'num_samples must be greater than or equal to 0.'
        self.label_key = label_key
        self.size = size
        self.pos_ratio = float(pos) / (float(pos) + float(neg))
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.centers = None

    def randomize(self, label, image):
        self.centers = generate_pos_neg_label_crop_centers(label, self.size, self.num_samples, self.pos_ratio,
                                                           image, self.image_threshold, self.R)

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        self.randomize(label, image)
        results = [dict() for _ in range(self.num_samples)]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.size)
                    results[i][key] = cropper(img)
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


class RandAffined(Randomizable, MapTransform):
    """
    A dictionary-based wrapper of :py:class:`monai.transforms.transforms.RandAffine`.
    """

    def __init__(self, keys,
                 spatial_size, prob=0.1,
                 rotate_range=None, shear_range=None, translate_range=None, scale_range=None,
                 mode='bilinear', padding_mode='zeros', as_tensor_output=True, device=None):
        """
        Args:
            keys (Hashable items): keys of the corresponding items to be transformed.
            spatial_size (list or tuple of int): output image spatial size.
                if ``data`` component has two spatial dimensions, ``spatial_size`` should have 2 elements [h, w].
                if ``data`` component has three spatial dimensions, ``spatial_size`` should have 3 elements [h, w, d].
            prob (float): probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            mode ('nearest'|'bilinear'): interpolation order. Defaults to ``'bilinear'``.
                if mode is a tuple of interpolation mode strings, each string corresponds to a key in ``keys``.
                this is useful to set different modes for different data items.
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices.
                Defaults to ``'zeros'``.
            as_tensor_output (bool): the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
        """
        MapTransform.__init__(self, keys)
        default_mode = 'bilinear' if isinstance(mode, (tuple, list)) else mode
        self.rand_affine = RandAffine(prob=prob,
                                      rotate_range=rotate_range, shear_range=shear_range,
                                      translate_range=translate_range, scale_range=scale_range,
                                      spatial_size=spatial_size,
                                      mode=default_mode, padding_mode=padding_mode,
                                      as_tensor_output=as_tensor_output, device=device)
        self.mode = mode

    def set_random_state(self, seed=None, state=None):
        self.rand_affine.set_random_state(seed, state)
        Randomizable.set_random_state(self, seed, state)
        return self

    def randomize(self):
        self.rand_affine.randomize()

    def __call__(self, data):
        d = dict(data)
        self.randomize()

        spatial_size = self.rand_affine.spatial_size
        if self.rand_affine.do_transform:
            grid = self.rand_affine.rand_affine_grid(spatial_size=spatial_size)
        else:
            grid = create_grid(spatial_size)

        if isinstance(self.mode, (tuple, list)):
            for key, m in zip(self.keys, self.mode):
                d[key] = self.rand_affine.resampler(d[key], grid, mode=m)
            return d

        for key in self.keys:  # same interpolation mode
            d[key] = self.rand_affine.resampler(d[key], grid, self.rand_affine.mode)
        return d


class Rand2DElasticd(Randomizable, MapTransform):
    """
    A dictionary-based wrapper of :py:class:`monai.transforms.transforms.Rand2DElastic`.
    """

    def __init__(self, keys,
                 spatial_size, spacing, magnitude_range, prob=0.1,
                 rotate_range=None, shear_range=None, translate_range=None, scale_range=None,
                 mode='bilinear', padding_mode='zeros', as_tensor_output=False, device=None):
        """
        Args:
            keys (Hashable items): keys of the corresponding items to be transformed.
            spatial_size (2 ints): specifying output image spatial size [h, w].
            spacing (2 ints): distance in between the control points.
            magnitude_range (2 ints): the random offsets will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob (float): probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            mode ('nearest'|'bilinear'): interpolation order. Defaults to ``'bilinear'``.
                if mode is a tuple of interpolation mode strings, each string corresponds to a key in ``keys``.
                this is useful to set different modes for different data items.
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices.
                Defaults to ``'zeros'``.
            as_tensor_output (bool): the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.
        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        MapTransform.__init__(self, keys)
        default_mode = 'bilinear' if isinstance(mode, (tuple, list)) else mode
        self.rand_2d_elastic = Rand2DElastic(spacing=spacing, magnitude_range=magnitude_range, prob=prob,
                                             rotate_range=rotate_range, shear_range=shear_range,
                                             translate_range=translate_range, scale_range=scale_range,
                                             spatial_size=spatial_size,
                                             mode=default_mode, padding_mode=padding_mode,
                                             as_tensor_output=as_tensor_output, device=device)
        self.mode = mode

    def set_random_state(self, seed=None, state=None):
        self.rand_2d_elastic.set_random_state(seed, state)
        Randomizable.set_random_state(self, seed, state)
        return self

    def randomize(self, spatial_size):
        self.rand_2d_elastic.randomize(spatial_size)

    def __call__(self, data):
        d = dict(data)
        spatial_size = self.rand_2d_elastic.spatial_size
        self.randomize(spatial_size)

        if self.rand_2d_elastic.do_transform:
            grid = self.rand_2d_elastic.deform_grid(spatial_size)
            grid = self.rand_2d_elastic.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(grid[None], spatial_size, mode='bicubic', align_corners=False)[0]
        else:
            grid = create_grid(spatial_size)

        if isinstance(self.mode, (tuple, list)):
            for key, m in zip(self.keys, self.mode):
                d[key] = self.rand_2d_elastic.resampler(d[key], grid, mode=m)
            return d

        for key in self.keys:  # same interpolation mode
            d[key] = self.rand_2d_elastic.resampler(d[key], grid, mode=self.rand_2d_elastic.mode)
        return d


class Rand3DElasticd(Randomizable, MapTransform):
    """
    A dictionary-based wrapper of :py:class:`monai.transforms.transforms.Rand3DElastic`.
    """

    def __init__(self, keys,
                 spatial_size, sigma_range, magnitude_range, prob=0.1,
                 rotate_range=None, shear_range=None, translate_range=None, scale_range=None,
                 mode='bilinear', padding_mode='zeros', as_tensor_output=False, device=None):
        """
        Args:
            keys (Hashable items): keys of the corresponding items to be transformed.
            spatial_size (3 ints): specifying output image spatial size [h, w, d].
            sigma_range (2 ints): a Gaussian kernel with standard deviation sampled
                 from ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range (2 ints): the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob (float): probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            mode ('nearest'|'bilinear'): interpolation order. Defaults to ``'bilinear'``.
                if mode is a tuple of interpolation mode strings, each string corresponds to a key in ``keys``.
                this is useful to set different modes for different data items.
            padding_mode ('zeros'|'border'|'reflection'): mode of handling out of range indices.
                Defaults to ``'zeros'``.
            as_tensor_output (bool): the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.
        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        MapTransform.__init__(self, keys)
        default_mode = 'bilinear' if isinstance(mode, (tuple, list)) else mode
        self.rand_3d_elastic = Rand3DElastic(sigma_range=sigma_range, magnitude_range=magnitude_range, prob=prob,
                                             rotate_range=rotate_range, shear_range=shear_range,
                                             translate_range=translate_range, scale_range=scale_range,
                                             spatial_size=spatial_size,
                                             mode=default_mode, padding_mode=padding_mode,
                                             as_tensor_output=as_tensor_output, device=device)
        self.mode = mode

    def set_random_state(self, seed=None, state=None):
        self.rand_3d_elastic.set_random_state(seed, state)
        Randomizable.set_random_state(self, seed, state)
        return self

    def randomize(self, grid_size):
        self.rand_3d_elastic.randomize(grid_size)

    def __call__(self, data):
        d = dict(data)
        spatial_size = self.rand_3d_elastic.spatial_size
        self.randomize(spatial_size)
        grid = create_grid(spatial_size)
        if self.rand_3d_elastic.do_transform:
            device = self.rand_3d_elastic.device
            grid = torch.tensor(grid).to(device)
            gaussian = GaussianFilter(spatial_dims=3, sigma=self.rand_3d_elastic.sigma, truncated=3., device=device)
            grid[:3] += gaussian(self.rand_3d_elastic.rand_offset[None])[0] * self.rand_3d_elastic.magnitude
            grid = self.rand_3d_elastic.rand_affine_grid(grid=grid)

        if isinstance(self.mode, (tuple, list)):
            for key, m in zip(self.keys, self.mode):
                d[key] = self.rand_3d_elastic.resampler(d[key], grid, mode=m)
            return d

        for key in self.keys:  # same interpolation mode
            d[key] = self.rand_3d_elastic.resampler(d[key], grid, mode=self.rand_3d_elastic.mode)
        return d


class Flipd(MapTransform):
    """Dictionary-based wrapper of Flip.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys (dict): Keys to pick data for transformation.
        spatial_axis (None, int or tuple of ints): Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, keys, spatial_axis=None):
        MapTransform.__init__(self, keys)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d


class RandFlipd(Randomizable, MapTransform):
    """Dict-based wrapper of RandFlip.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob (float): Probability of flipping.
        spatial_axis (None, int or tuple of ints): Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, keys, prob=0.1, spatial_axis=None):
        MapTransform.__init__(self, keys)
        self.spatial_axis = spatial_axis
        self.prob = prob

        self._do_transform = False
        self.flipper = Flip(spatial_axis=spatial_axis)

    def randomize(self):
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d


class Rotated(MapTransform):
    """Dictionary-based wrapper of Rotate.

    Args:
        keys (dict): Keys to pick data for transformation.
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

    def __init__(self, keys, angle, spatial_axes=(0, 1), reshape=True, order=1,
                 mode='constant', cval=0, prefilter=True):
        MapTransform.__init__(self, keys)
        self.rotator = Rotate(angle=angle, spatial_axes=spatial_axes, reshape=reshape,
                              order=order, mode=mode, cval=cval, prefilter=prefilter)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.rotator(d[key])
        return d


class RandRotated(Randomizable, MapTransform):
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

    def __init__(self, keys, degrees, prob=0.1, spatial_axes=(0, 1), reshape=True, order=1,
                 mode='constant', cval=0, prefilter=True):
        MapTransform.__init__(self, keys)
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

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        rotator = Rotate(self.angle, self.spatial_axes, self.reshape, self.order,
                         self.mode, self.cval, self.prefilter)
        for key in self.keys:
            d[key] = rotator(d[key])
        return d


class Zoomd(MapTransform):
    """Dictionary-based wrapper of Zoom transform.

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

    def __init__(self, keys, zoom, order=3, mode='constant', cval=0,
                 prefilter=True, use_gpu=False, keep_size=False):
        MapTransform.__init__(self, keys)
        self.zoomer = Zoom(zoom=zoom, order=order, mode=mode, cval=cval,
                           prefilter=prefilter, use_gpu=use_gpu, keep_size=keep_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.zoomer(d[key])
        return d


class RandZoomd(Randomizable, MapTransform):
    """Dict-based wrapper of RandZoom.

    Args:
        keys (dict): Keys to pick data for transformation.
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

    def __init__(self, keys, prob=0.1, min_zoom=0.9,
                 max_zoom=1.1, order=3, mode='constant',
                 cval=0, prefilter=True, use_gpu=False, keep_size=False):
        MapTransform.__init__(self, keys)
        if hasattr(min_zoom, '__iter__') and \
           hasattr(max_zoom, '__iter__'):
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

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        zoomer = Zoom(self._zoom, self.order, self.mode, self.cval, self.prefilter, self.use_gpu, self.keep_size)
        for key in self.keys:
            d[key] = zoomer(d[key])
        return d


class DeleteKeysd(MapTransform):
    """
    Delete specified keys from data dictionary to release memory.
    It will remove the key-values and copy the others to construct a new dictionary.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        return {key: val for key, val in data.items() if key not in self.keys}


SpacingD = SpacingDict = Spacingd
OrientationD = OrientationDict = Orientationd
LoadNiftiD = LoadNiftiDict = LoadNiftid
LoadPNGD = LoadPNGDict = LoadPNGd
AsChannelFirstD = AsChannelFirstDict = AsChannelFirstd
AddChannelD = AddChannelDict = AddChanneld
ToTensorD = ToTensorDict = ToTensord
Rotate90D = Rotate90Dict = Rotate90d
RescaleD = RescaleDict = Rescaled
ResizeD = ResizeDict = Resized
RandUniformPatchD = RandUniformPatchDict = RandUniformPatchd
RandRotate90D = RandRotate90Dict = RandRotate90d
NormalizeIntensityD = NormalizeIntensityDict = NormalizeIntensityd
ScaleIntensityRangeD = ScaleIntensityRangeDict = ScaleIntensityRanged
RandCropByPosNegLabelD = RandCropByPosNegLabelDict = RandCropByPosNegLabeld
RandAffineD = RandAffineDict = RandAffined
Rand2DElasticD = Rand2DElasticDict = Rand2DElasticd
Rand3DElasticD = Rand3DElasticDict = Rand3DElasticd
FlipD = FlipDict = Flipd
RandFlipD = RandFlipDict = RandFlipd
RotateD = RotateDict = Rotated
RandRotateD = RandRotateDict = RandRotated
ZoomD = ZoomDict = Zoomd
RandZoomD = RandZoomDict = RandZoomd
DeleteKeysD = DeleteKeysDict = DeleteKeysd
