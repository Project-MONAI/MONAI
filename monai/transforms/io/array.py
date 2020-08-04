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
A collection of "vanilla" transforms for IO functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import Optional

import numpy as np
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai.config import KeysCollection
from monai.data.utils import correct_nifti_header_if_necessary
from monai.transforms.compose import Transform
from monai.utils import ensure_tuple, optional_import

nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")


class LoadNifti(Transform):
    """
    Load Nifti format file or files from provided path. If loading a list of
    files, stack them together and add a new dimension as first dimension, and
    use the meta data of the first image to represent the stacked result. Note
    that the affine transform of all the images should be same if ``image_only=False``.
    """

    def __init__(
        self, as_closest_canonical: bool = False, image_only: bool = False, dtype: Optional[np.dtype] = np.float32
    ) -> None:
        """
        Args:
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            image_only: if True return only the image volume, otherwise return image data array and header dict.
            dtype: if not None convert the loaded image to this data type.

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
            header["filename_or_obj"] = name
            header["affine"] = img.affine
            header["original_affine"] = img.affine.copy()
            header["as_closest_canonical"] = self.as_closest_canonical
            ndim = img.header["dim"][0]
            spatial_rank = min(ndim, 3)
            header["spatial_shape"] = img.header["dim"][1 : spatial_rank + 1]

            if self.as_closest_canonical:
                img = nib.as_closest_canonical(img)
                header["affine"] = img.affine

            img_array.append(np.array(img.get_fdata(dtype=self.dtype)))
            img.uncache()

            if self.image_only:
                continue

            if not compatible_meta:
                for meta_key in header:
                    meta_datum = header[meta_key]
                    if (
                        isinstance(meta_datum, np.ndarray)
                        and np_str_obj_array_pattern.search(meta_datum.dtype.str) is not None
                    ):
                        continue
                    compatible_meta[meta_key] = meta_datum
            else:
                assert np.allclose(
                    header["affine"], compatible_meta["affine"]
                ), "affine data of all images should be same."

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        if self.image_only:
            return img_array
        return img_array, compatible_meta


class LoadPNG(Transform):
    """
    Load common 2D image format (PNG, JPG, etc. using PIL) file or files from provided path.
    If loading a list of files, stack them together and add a new dimension as first dimension,
    and use the meta data of the first image to represent the stacked result.
    It's based on the Image module in PIL library:
    https://pillow.readthedocs.io/en/stable/reference/Image.html
    """

    def __init__(self, image_only: bool = False, dtype: Optional[np.dtype] = np.float32) -> None:
        """
        Args:
            image_only: if True return only the image volume, otherwise return image data array and metadata.
            dtype: if not None convert the loaded image to this data type.
        """
        self.image_only = image_only
        self.dtype = dtype

    def __call__(self, filename):
        """
        Args:
            filename (str, list, tuple, file): path file or file-like object or a list of files.
        """
        filename = ensure_tuple(filename)
        img_array = list()
        compatible_meta = None
        for name in filename:
            img = Image.open(name)
            data = np.asarray(img)
            if self.dtype:
                data = data.astype(self.dtype)
            img_array.append(data)

            if self.image_only:
                continue

            meta = dict()
            meta["filename_or_obj"] = name
            meta["spatial_shape"] = data.shape[:2]
            meta["format"] = img.format
            meta["mode"] = img.mode
            meta["width"] = img.width
            meta["height"] = img.height
            meta["info"] = img.info
            if not compatible_meta:
                compatible_meta = meta
            else:
                assert np.allclose(
                    meta["spatial_shape"], compatible_meta["spatial_shape"]
                ), "all the images in the list should have same spatial shape."

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array if self.image_only else (img_array, compatible_meta)


class LoadNumpy(Transform):
    """
    Load arrays or pickled objects from .npy, .npz or pickled files, file or files are from provided path.
    A typical usage is to load the `mask` data for classification task.
    If loading a list of files or laoding npz file, stack results together and add a new dimension as first dimension,
    and use the meta data of the first file to represent the stacked result.
    It can load part of the npz file with specified `npz_keys`.
    It's based on the Numpy load/read API:
    https://numpy.org/doc/stable/reference/generated/numpy.load.html

    """

    def __init__(
        self, data_only: bool = False, dtype: Optional[np.dtype] = np.float32, npz_keys: Optional[KeysCollection] = None
    ) -> None:
        """
        Args:
            data_only: if True return only the data array, otherwise return data array and metadata.
            dtype: if not None convert the loaded data to this data type.
            npz_keys: if loading npz file, only load the specified keys, if None, load all the items.
                stack the loaded items together to construct a new first dimension.

        """
        self.data_only = data_only
        self.dtype = dtype
        if npz_keys is not None:
            npz_keys = ensure_tuple(npz_keys)
        self.npz_keys = npz_keys

    def __call__(self, filename):
        """
        Args:
            filename (str, list, tuple, file): path file or file-like object or a list of files.

        Raises:
            ValueError: When ``filename`` is a sequence and contains a "npz" file extension.

        """
        if isinstance(filename, (tuple, list)):
            for name in filename:
                if name.endswith(".npz"):
                    raise ValueError("Cannot load a sequence of npz files.")
        filename = ensure_tuple(filename)
        data_array = list()
        compatible_meta = None

        def _save_data_meta(data_array, name, data, compatible_meta):
            data_array.append(data if self.dtype is None else data.astype(self.dtype))
            if not self.data_only:
                meta = dict()
                meta["filename_or_obj"] = name
                meta["spatial_shape"] = data.shape
                if not compatible_meta:
                    compatible_meta = meta
                else:
                    assert np.allclose(
                        meta["spatial_shape"], compatible_meta["spatial_shape"]
                    ), "all the data in the list should have same shape."
            return compatible_meta

        for name in filename:
            data = np.load(name, allow_pickle=True)
            if name.endswith(".npz"):
                # load expected items from NPZ file
                npz_keys = [f"arr_{i}" for i in range(len(data))] if self.npz_keys is None else self.npz_keys
                for k in npz_keys:
                    compatible_meta = _save_data_meta(data_array, name, data[k], compatible_meta)
            else:
                compatible_meta = _save_data_meta(data_array, name, data, compatible_meta)

        data_array = np.stack(data_array, axis=0) if len(data_array) > 1 else data_array[0]
        return data_array if self.data_only else (data_array, compatible_meta)
