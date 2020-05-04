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
A collection of dictionary-based wrappers around the "vanilla" transforms for IO functions
defined in :py:class:`monai.transforms.io.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

import numpy as np

from monai.transforms.compose import MapTransform
from monai.transforms.io.array import LoadNifti, LoadPNG


class LoadNiftid(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadNifti`,
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
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadPNG`.
    """

    def __init__(self, keys, dtype=np.float32, meta_key_format='{}.{}'):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.
            meta_key_format (str): key format to store meta data of the loaded image.
                it must contain 2 fields for the key of this image and the key of every meta data item.
        """
        super().__init__(keys)
        self.loader = LoadPNG(False, dtype)
        self.meta_key_format = meta_key_format

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            data = self.loader(d[key])
            assert isinstance(data, (tuple, list)), 'loader must return a tuple or list.'
            d[key] = data[0]
            assert isinstance(data[1], dict), 'metadata must be a dict.'
            for k in sorted(data[1]):
                key_to_add = self.meta_key_format.format(key, k)
                d[key_to_add] = data[1][k]
        return d


LoadNiftiD = LoadNiftiDict = LoadNiftid
LoadPNGD = LoadPNGDict = LoadPNGd
