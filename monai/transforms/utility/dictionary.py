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
A collection of dictionary-based wrappers around the "vanilla" transforms for utility functions
defined in :py:class:`monai.transforms.transforms`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import numpy as np

from monai.transforms.compose import MapTransform
from monai.transforms.utility.array import AddChannel, AsChannelFirst, LoadNifti, ToTensor, \
    LoadPNG, AsChannelLast, CastToType, RepeatChannel, SqueezeDim


class LoadNiftid(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transfroms.LoadNifti`,
    must load image and metadata together. If loading a list of files in one key,
    stack them together and add a new dimension as the first dimension, and use the
    meta data of the first image to represent the stacked result. Note that the affine
    transform of all the stacked images should be same. The output metadata field will
    be created as ``self.meta_key_format(key, metadata_key)``.
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
        super().__init__(keys)
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
    Dictionary-based wrapper of :py:class:`monai.transforms.transfroms.LoadPNG`.
    """

    def __init__(self, keys, dtype=np.float32):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.
        """
        super().__init__(keys)
        self.loader = LoadPNG(dtype)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.loader(d[key])
        return d


class AsChannelFirstd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transfroms.AsChannelFirst`.
    """

    def __init__(self, keys, channel_dim=-1):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim (int): which dimension of input image is the channel, default is the last dimension.
        """
        super().__init__(keys)
        self.converter = AsChannelFirst(channel_dim=channel_dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AsChannelLastd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transfroms.AsChannelLast`.
    """

    def __init__(self, keys, channel_dim=0):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            channel_dim (int): which dimension of input image is the channel, default is the first dimension.
        """
        super().__init__(keys)
        self.converter = AsChannelLast(channel_dim=channel_dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transfroms.AddChannel`.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.adder = AddChannel()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.adder(d[key])
        return d


class RepeatChanneld(MapTransform):
    """
    dictionary-based wrapper of :py:class:`monai.transforms.transforms.RepeatChannel`.
    """

    def __init__(self, keys, repeats):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            repeats (int): the number of repetitions for each element.
        """
        super().__init__(keys)
        self.repeater = RepeatChannel(repeats)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.repeater(d[key])
        return d


class CastToTyped(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transfroms.CastToType`.
    """

    def __init__(self, keys, dtype=np.float32):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype (np.dtype): convert image to this data type, default is `np.float32`.
        """
        MapTransform.__init__(self, keys)
        self.converter = CastToType(dtype)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class ToTensord(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transfroms.ToTensor`.
    """

    def __init__(self, keys):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.converter = ToTensor()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
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
        super().__init__(keys)

    def __call__(self, data):
        return {key: val for key, val in data.items() if key not in self.keys}


class SqueezeDimd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.transforms.SqueezeDim`.
    """

    def __init__(self, keys, dim=None):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dim (int): dimension to be squeezed.
                Default: None (all dimensions of size 1 will be removed)
        """
        super().__init__(keys)
        self.converter = SqueezeDim(dim=dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


LoadNiftiD = LoadNiftiDict = LoadNiftid
LoadPNGD = LoadPNGDict = LoadPNGd
AsChannelFirstD = AsChannelFirstDict = AsChannelFirstd
AsChannelLastD = AsChannelLastDict = AsChannelLastd
AddChannelD = AddChannelDict = AddChanneld
RepeatChannelD = RepeatChannelDict = RepeatChanneld
CastToTypeD = CastToTypeDict = CastToTyped
ToTensorD = ToTensorDict = ToTensord
DeleteKeysD = DeleteKeysDict = DeleteKeysd
SqueezeDimD = SqueezeDimDict = SqueezeDimd
