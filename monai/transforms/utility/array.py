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
A collection of "vanilla" transforms for utility functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import numpy as np
import nibabel as nib
from PIL import Image
import torch
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai.data.utils import correct_nifti_header_if_necessary
from monai.transforms.compose import Transform
from monai.utils.misc import ensure_tuple


class LoadNifti(Transform):
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
            - header['original_affine'] will be additionally created to store the original affine.
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
            header['original_affine'] = img.affine.copy()
            header['as_closest_canonical'] = self.as_closest_canonical
            ndim = img.header['dim'][0]
            spatial_rank = min(ndim, 3)
            header['spatial_shape'] = img.header['dim'][1:spatial_rank + 1]

            if self.as_closest_canonical:
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


class LoadPNG(Transform):
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


class AsChannelFirst(Transform):
    """
    Change the channel dimension of the image to the first dimension.

    Most of the image transformations in ``monai.transforms``
    assume the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used to convert, for example, a channel-last image array in shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels) into the channel-first format,
    so that the multidimensional image array can be correctly interpreted by the other transforms.

    Args:
        channel_dim (int): which dimension of input image is the channel, default is the last dimension.
    """

    def __init__(self, channel_dim=-1):
        assert isinstance(channel_dim, int) and channel_dim >= -1, 'invalid channel dimension.'
        self.channel_dim = channel_dim

    def __call__(self, img):
        return np.moveaxis(img, self.channel_dim, 0)


class AsChannelLast(Transform):
    """
    Change the channel dimension of the image to the last dimension.

    Some of other 3rd party transforms assume the input image is in the channel-last format with shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels).

    This transform could be used to convert, for example, a channel-first image array in shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]) into the channel-last format,
    so that MONAI transforms can construct a chain with other 3rd party transforms together.

    Args:
        channel_dim (int): which dimension of input image is the channel, default is the first dimension.
    """

    def __init__(self, channel_dim=0):
        assert isinstance(channel_dim, int) and channel_dim >= -1, 'invalid channel dimension.'
        self.channel_dim = channel_dim

    def __call__(self, img):
        return np.moveaxis(img, self.channel_dim, -1)


class AddChannel(Transform):
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


class RepeatChannel(Transform):
    """
    Repeat channel data to construct expected input shape for models.
    The `repeats` count includes the origin data, for example:
    ``RepeatChannel(repeats=2)([[1, 2], [3, 4]])`` generates: ``[[1, 2], [1, 2], [3, 4], [3, 4]]``

    Args:
        repeats (int): the number of repetitions for each element.
    """

    def __init__(self, repeats):
        assert repeats > 0, 'repeats count must be greater than 0.'
        self.repeats = repeats

    def __call__(self, img):
        return np.repeat(img, self.repeats, 0)


class CastToType(Transform):
    """
    Cast the image data to specified numpy data type.
    """

    def __init__(self, dtype=np.float32):
        """
        Args:
            dtype (np.dtype): convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(self, img):
        assert isinstance(img, np.ndarray), 'image must be numpy array.'
        return img.astype(self.dtype)


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    """

    def __call__(self, img):
        if torch.is_tensor(img):
            return img.contiguous()
        return torch.as_tensor(np.ascontiguousarray(img))


class Transpose(Transform):
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    def __init__(self, indices):
        self.indices = indices

    def __call__(self, img):
        return img.transpose(self.indices)


class SqueezeDim(Transform):
    """
    Squeeze undesired unitary dimensions
    """

    def __init__(self, dim=None):
        """
        Args:
            dim (int): dimension to be squeezed.
                Default: None (all dimensions of size 1 will be removed)
        """
        if dim is not None:
            assert isinstance(dim, int) and dim >= -1, 'invalid channel dimension.'
        self.dim = dim

    def __call__(self, img):
        """
        Args:
            img (ndarray): numpy arrays with required dimension `dim` removed
        """
        return np.squeeze(img, self.dim)
