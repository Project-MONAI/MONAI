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

from typing import Optional

import numpy as np

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from monai.transforms.io.array import LoadNifti, LoadPNG


class LoadNiftid(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadNifti`,
    must load image and metadata together. If loading a list of files in one key,
    stack them together and add a new dimension as the first dimension, and use the
    meta data of the first image to represent the stacked result. Note that the affine
    transform of all the stacked images should be same. The output metadata field will
    be created as ``key_{meta_key_postfix}``.
    """

    def __init__(
        self,
        keys: KeysCollection,
        as_closest_canonical: bool = False,
        dtype: Optional[np.dtype] = np.float32,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.
            meta_key_postfix: use `key_{postfix}` to to store meta data of the nifti image,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting (bool): whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.

        Raises:
            ValueError: meta_key_postfix must be a string.

        """
        super().__init__(keys)
        self.loader = LoadNifti(as_closest_canonical, False, dtype)
        if not isinstance(meta_key_postfix, str):
            raise ValueError("meta_key_postfix must be a string.")
        self.meta_key_postfix = meta_key_postfix
        self.overwriting = overwriting

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            data = self.loader(d[key])
            assert isinstance(data, (tuple, list)), "loader must return a tuple or list."
            d[key] = data[0]
            assert isinstance(data[1], dict), "metadata must be a dict."
            key_to_add = f"{key}_{self.meta_key_postfix}"
            if key_to_add in d and not self.overwriting:
                raise KeyError(f"meta data with key {key_to_add} already exists.")
            d[key_to_add] = data[1]
        return d


class LoadPNGd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadPNG`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Optional[np.dtype] = np.float32,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype (np.dtype, optional): if not None convert the loaded image to this data type.
            meta_key_postfix: use `key_{postfix}` to to store meta data of the nifti image,
                default is `meta_dict`. The meta data is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow to overwrite existing meta data of same key.
                default is False, which will raise exception if encountering existing key.

        Raises:
            ValueError: meta_key_postfix must be a string.

        """
        super().__init__(keys)
        self.loader = LoadPNG(False, dtype)
        if not isinstance(meta_key_postfix, str):
            raise ValueError("meta_key_postfix must be a string.")
        self.meta_key_postfix = meta_key_postfix
        self.overwriting = overwriting

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            data = self.loader(d[key])
            assert isinstance(data, (tuple, list)), "loader must return a tuple or list."
            d[key] = data[0]
            assert isinstance(data[1], dict), "metadata must be a dict."
            key_to_add = f"{key}_{self.meta_key_postfix}"
            if key_to_add in d and not self.overwriting:
                raise KeyError(f"meta data with key {key_to_add} already exists.")
            d[key_to_add] = data[1]
        return d


LoadNiftiD = LoadNiftiDict = LoadNiftid
LoadPNGD = LoadPNGDict = LoadPNGd
