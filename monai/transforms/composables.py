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
defined in `monai.transforms.transforms`.
"""

import torch
from collections.abc import Hashable

import monai
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.compose import Randomizable, Transform
from monai.transforms.transforms import (LoadNifti, AsChannelFirst, Orientation,
                                         AddChannel, Spacing, Rotate90, SpatialCrop,
                                         RandAffine, Rand2DElastic, Rand3DElastic,
                                         Flip, Rotate, Zoom)
from monai.utils.misc import ensure_tuple
from monai.transforms.utils import generate_pos_neg_label_crop_centers, create_grid
from monai.utils.aliases import alias

export = monai.utils.export("monai.transforms")


@export
class MapTransform(Transform):
    """
    A subclass of ``monai.transforms.compose.Transform`` with an assumption
    that the ``data`` input of ``self.__call__`` is a MutableMapping such as ``dict``.

    The ``keys`` parameter will be used to get and set the actual data
    item to transform.  That is, the callable of this transform should
    follow the pattern:
    .. code-block:: python

        def __call__(self, data):
            for key in self.keys:
                if key in data:
                    # update output data with some_transform_function(data[key]).
                else:
                    # do nothing or some exceptions handling.
            return data
    """

    def __init__(self, keys):
        self.keys = ensure_tuple(keys)
        if not self.keys:
            raise ValueError('keys unspecified')
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise ValueError('keys should be a hashable or a sequence of hashables, got {}'.format(type(key)))


@export
@alias('SpacingD', 'SpacingDict')
class Spacingd(MapTransform):
    """
    dictionary-based wrapper of :class: `monai.transforms.transforms.Spacing`.
    """

    def __init__(self, keys, affine_key, pixdim, interp_order=2, keep_shape=False, output_key='spacing'):
        """
        Args:
            affine_key (hashable): the key to the original affine.
                The affine will be used to compute input data's pixdim.
            pixdim (sequence of floats): output voxel spacing.
            interp_order (int or sequence of ints): int: the same interpolation order
                for all data indexed by `self,keys`; sequence of ints, should
                correspond to an interpolation order for each data item indexed
                by `self.keys` respectively.
            keep_shape (bool): whether to maintain the original spatial shape
                after resampling. Defaults to False.
            output_key (hashable): key to be added to the output dictionary to track
                the pixdim status.

        """
        MapTransform.__init__(self, keys)
        self.affine_key = affine_key
        self.spacing_transform = Spacing(pixdim, keep_shape=keep_shape)
        interp_order = ensure_tuple(interp_order)
        self.interp_order = interp_order \
            if len(interp_order) == len(self.keys) else interp_order * len(self.keys)
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        affine = d[self.affine_key]
        original_pixdim, new_pixdim = None, None
        for key, interp in zip(self.keys, self.interp_order):
            d[key], original_pixdim, new_pixdim = self.spacing_transform(d[key], affine, interp_order=interp)
        d[self.output_key] = {'original_pixdim': original_pixdim, 'current_pixdim': new_pixdim}
        return d


@export
@alias('OrientationD', 'OrientationDict')
class Orientationd(MapTransform):
    """
    dictionary-based wrapper of :class: `monai.transforms.transforms.Orientation`.
    """

    def __init__(self, keys, affine_key, axcodes, labels=None, output_key='orientation'):
        """
        Args:
            affine_key (hashable): the key to the original affine.
                The affine will be used to compute input data's orientation.
            axcodes (N elements sequence): for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                    (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            labels : optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                see: ``nibabel.orientations.ornt2axcodes``.
        """
        MapTransform.__init__(self, keys)
        self.affine_key = affine_key
        self.orientation_transform = Orientation(axcodes=axcodes, labels=labels)
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        affine = d[self.affine_key]
        original_ornt, new_ornt = None, None
        for key in self.keys:
            d[key], original_ornt, new_ornt = self.orientation_transform(d[key], affine)
        d[self.output_key] = {'original_ornt': original_ornt, 'current_ornt': new_ornt}
        return d


@export
@alias('LoadNiftiD', 'LoadNiftiDict')
class LoadNiftid(MapTransform):
    """
    dictionary-based wrapper of LoadNifti, must load image and metadata together.
    """

    def __init__(self, keys, as_closest_canonical=False, dtype=None, meta_key_format='{}.{}', overwriting_keys=False):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: monai.transform.composables.MapTransform
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
            assert isinstance(data, (tuple, list)), 'if data contains metadata, must be tuple or list.'
            d[key] = data[0]
            assert isinstance(data[1], dict), 'metadata must be in dict format.'
            for k in sorted(data[1].keys()):
                key_to_add = self.meta_key_format.format(key, k)
                if key_to_add in d and self.overwriting_keys is False:
                    raise KeyError('meta data key is alreay existing.')
                d[key_to_add] = data[1][k]
        return d


@export
@alias('AsChannelFirstD', 'AsChannelFirstDict')
class AsChannelFirstd(MapTransform):
    """
    dictionary-based wrapper of AsChannelFirst.
    """

    def __init__(self, keys, channel_dim=-1):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: monai.transform.composables.MapTransform
            channel_dim (int): which dimension of input image is the channel, default is the last dimension.
        """
        MapTransform.__init__(self, keys)
        self.converter = AsChannelFirst(channel_dim=channel_dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


@export
@alias('AddChannelD', 'AddChannelDict')
class AddChanneld(MapTransform):
    """
    dictionary-based wrapper of AddChannel.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: monai.transform.composables.MapTransform
        """
        MapTransform.__init__(self, keys)
        self.adder = AddChannel()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.adder(d[key])
        return d


@export
@alias('Rotate90D', 'Rotate90Dict')
class Rotate90d(MapTransform):
    """
    dictionary-based wrapper of Rotate90.
    """

    def __init__(self, keys, k=1, axes=(1, 2)):
        """
        Args:
            k (int): number of times to rotate by 90 degrees.
            axes (2 ints): defines the plane to rotate with 2 axes.
        """
        MapTransform.__init__(self, keys)
        self.k = k
        self.plane_axes = axes

        self.rotator = Rotate90(self.k, self.plane_axes)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.rotator(d[key])
        return d


@export
@alias('UniformRandomPatchD', 'UniformRandomPatchDict')
class UniformRandomPatchd(Randomizable, MapTransform):
    """
    Selects a patch of the given size chosen at a uniformly random position in the image.
    """

    def __init__(self, keys, patch_size):
        MapTransform.__init__(self, keys)

        self.patch_size = (None,) + tuple(patch_size)

        self._slices = None

    def randomize(self, image_shape, patch_shape):
        self._slices = get_random_patch(image_shape, patch_shape, self.R)

    def __call__(self, data):
        d = dict(data)

        image_shape = d[self.keys[0]].shape  # image shape from the first data key
        patch_size = get_valid_patch_size(image_shape, self.patch_size)
        self.randomize(image_shape, patch_size)
        for key in self.keys:
            d[key] = d[key][self._slices]
        return d


@export
@alias('RandRotate90D', 'RandRotate90Dict')
class RandRotate90d(Randomizable, MapTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `axes`.
    """

    def __init__(self, keys, prob=0.1, max_k=3, axes=(1, 2)):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: monai.transform.composables.MapTransform
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            axes (2 ints): defines the plane to rotate with 2 axes.
                (Default to (1, 2))
        """
        MapTransform.__init__(self, keys)

        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.axes = axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self):
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        self.randomize()
        if not self._do_transform:
            return data

        rotator = Rotate90(self._rand_k, self.axes)
        d = dict(data)
        for key in self.keys:
            d[key] = rotator(d[key])
        return d


@export
@alias('RandCropByPosNegLabelD', 'RandCropByPosNegLabelDict')
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
    """

    def __init__(self, keys, label_key, size, pos=1, neg=1, num_samples=1):
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
        self.centers = None

    def randomize(self, label):
        self.centers = generate_pos_neg_label_crop_centers(label, self.size, self.num_samples, self.pos_ratio, self.R)

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        self.randomize(label)
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


@export
@alias('RandAffineD', 'RandAffineDict')
class RandAffined(Randomizable, MapTransform):
    """
    A dictionary-based wrapper of ``monai.transforms.transforms.RandAffine``.
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
            - ``monai.transform.composables.MapTransform``
            - ``RandAffineGrid`` for the random affine paramters configurations.
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


@export
@alias('Rand2DElasticD', 'Rand2DElasticDict')
class Rand2DElasticd(Randomizable, MapTransform):
    """
    A dictionary-based wrapper of ``monai.transforms.transforms.Rand2DElastic``.
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
            - ``RandAffineGrid`` for the random affine paramters configurations.
            - ``Affine`` for the affine transformation parameters configurations.
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


@export
@alias('Rand3DElasticD', 'Rand3DElasticDict')
class Rand3DElasticd(Randomizable, MapTransform):
    """
    A dictionary-based wrapper of ``monai.transforms.transforms.Rand3DElastic``.
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
            - ``RandAffineGrid`` for the random affine paramters configurations.
            - ``Affine`` for the affine transformation parameters configurations.
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


@export
@alias('FlipD', 'FlipDict')
class Flipd(MapTransform):
    """Dictionary-based wrapper of Flip.

    Args:
        keys (dict): Keys to pick data for transformation.
        axis (None, int or tuple of ints): Axes along which to flip over. Default is None.
    """

    def __init__(self, keys, axis=None):
        MapTransform.__init__(self, keys)
        self.flipper = Flip(axis=axis)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d


@export
@alias('RandFlipD', 'RandFlipDict')
class RandFlipd(Randomizable, MapTransform):
    """Dict-based wrapper of RandFlip.

    Args:
        prob (float): Probability of flipping.
        axis (None, int or tuple of ints): Axes along which to flip over. Default is None.
    """

    def __init__(self, keys, prob=0.1, axis=None):
        MapTransform.__init__(self, keys)
        self.axis = axis
        self.prob = prob

        self._do_transform = False
        self.flipper = Flip(axis=axis)

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


@export
@alias('RotateD', 'RotateDict')
class Rotated(MapTransform):
    """Dictionary-based wrapper of Rotate.

    Args:
        keys (dict): Keys to pick data for transformation.
        angle (float): Rotation angle in degrees.
        axes (tuple of 2 ints): Axes of rotation. Default: (1, 2). This is the first two
            axis in spatial dimensions according to MONAI channel first shape assumption.
        reshape (bool): If true, output shape is made same as input. Default: True.
        order (int): Order of spline interpolation. Range 0-5. Default: 1. This is
            different from scipy where default interpolation is 3.
        mode (str): Points outside boundary filled according to this mode. Options are
            'constant', 'nearest', 'reflect', 'wrap'. Default: 'constant'.
        cval (scalar): Values to fill outside boundary. Default: 0.
        prefiter (bool): Apply spline_filter before interpolation. Default: True.
    """

    def __init__(self, keys, angle, axes=(1, 2), reshape=True, order=1,
                 mode='constant', cval=0, prefilter=True):
        MapTransform.__init__(self, keys)
        self.rotator = Rotate(angle=angle, axes=axes, reshape=reshape,
                              order=order, mode=mode, cval=cval, prefilter=prefilter)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.rotator(d[key])
        return d


@export
@alias('RandRotateD', 'RandRotateDict')
class RandRotated(Randomizable, MapTransform):
    """Randomly rotates the input arrays.

    Args:
        prob (float): Probability of rotation.
        degrees (tuple of float or float): Range of rotation in degrees. If single number,
            angle is picked from (-degrees, degrees). 
        axes (tuple of 2 ints): Axes of rotation. Default: (1, 2). This is the first two
            axis in spatial dimensions according to MONAI channel first shape assumption.
        reshape (bool): If true, output shape is made same as input. Default: True.
        order (int): Order of spline interpolation. Range 0-5. Default: 1. This is
            different from scipy where default interpolation is 3.
        mode (str): Points outside boundary filled according to this mode. Options are 
            'constant', 'nearest', 'reflect', 'wrap'. Default: 'constant'.
        cval (scalar): Value to fill outside boundary. Default: 0.
        prefiter (bool): Apply spline_filter before interpolation. Default: True.
    """
    def __init__(self, keys, degrees, prob=0.1, axes=(1, 2), reshape=True, order=1, 
                 mode='constant', cval=0, prefilter=True):
        MapTransform.__init__(self, keys)
        self.prob = prob
        self.degrees = degrees
        self.reshape = reshape
        self.order = order
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.axes = axes

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
        rotator = Rotate(self.angle, self.axes, self.reshape, self.order,
                         self.mode, self.cval, self.prefilter)
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d


@export
@alias('ZoomD', 'ZoomDict')
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


@export
@alias('RandZoomD', 'RandZoomDict')
class RandZoomd(Randomizable, MapTransform):
    """Dict-based wrapper of RandZoom.

    Args:
        keys (dict): Keys to pick data for transformation.
        prob (float): Probability of zooming.
        min_zoom (float or sequence): Min zoom factor. Can be float or sequence same size as image.
        max_zoom (float or sequence): Max zoom factor. Can be float or sequence same size as image.
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
