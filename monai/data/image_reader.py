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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import KeysCollection
from monai.data.utils import correct_nifti_header_if_necessary
from monai.utils import ensure_tuple, optional_import

from .utils import is_supported_format

if TYPE_CHECKING:
    import itk  # type: ignore
    import nibabel as nib
    from itk import Image  # type: ignore
    from nibabel.nifti1 import Nifti1Image
else:
    itk, _ = optional_import("itk", allow_namespace_pkg=True)
    Image, _ = optional_import("itk", allow_namespace_pkg=True, name="Image")
    nib, _ = optional_import("nibabel")
    Nifti1Image, _ = optional_import("nibabel.nifti1", name="Nifti1Image")


class ImageReader(ABC):
    """Abstract class to define interface APIs to load image files.
    users need to call `read` to load image and then use `get_data`
    to get the image data and properties from meta data.

    """

    @abstractmethod
    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        Verify whether the specified file or files format is supported by current reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the subffixes.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def read(self, data: Union[Sequence[str], str, Any], **kwargs) -> Union[Sequence[Any], Any]:
        """
        Read image data from specified file or files, or set a loaded image.
        Note that it returns the raw data, so different readers return different image data type.

        Args:
            data: file name or a list of file names to read, or a loaded image object.
            kwargs: additional args for 3rd party reader API.


        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, Dict]:
        """
        Extract data array and meta data from loaded image and return them.
        This function must return 2 objects, first is numpy array of image data, second is dict of meta data.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ITKReader(ImageReader):
    """
    Load medical images based on ITK library.
    All the supported image formats can be found:
    https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/IO

    Args:
        c_order_axis_indexing: if `True`, the numpy array will have C-order indexing.
            this is the reverse of how indices are specified in ITK, i.e. k,j,i versus i,j,k.
            however C-order indexing is expected by most algorithms in numpy / scipy.

    """

    def __init__(self, c_order_axis_indexing: bool = False):
        super().__init__()
        self._img: Optional[Sequence[Image]] = None
        self.c_order_axis_indexing = c_order_axis_indexing

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        Verify whether the specified file or files format is supported by ITK reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the subffixes.

        """
        return True

    def read(self, data: Union[Sequence[str], str, Image], **kwargs):
        """
        Read image data from specified file or files, or set a `itk.Image` object.
        Note that the returned object is ITK image object or list of ITK image objects.
        `self._img` is always a list, even only has 1 image.

        Args:
            data: file name or a list of file names to read, or a `itk.Image` object for the usage that
                the image data is already in memory and no need to read from file again.
            kwargs: additional args for `itk.imread` API. more details about available args:
                https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itkExtras.py

        """
        self._img = list()
        if isinstance(data, Image):
            self._img.append(data)
            return data

        filenames: Sequence[str] = ensure_tuple(data)
        for name in filenames:
            self._img.append(itk.imread(name, **kwargs))
        return self._img if len(filenames) > 1 else self._img[0]

    def get_data(self):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns 2 objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores in meta dict.
        If loading a list of files, stack them together and add a new dimension as first dimension,
        and use the meta data of the first image to represent the stacked result.

        """
        img_array: List[np.ndarray] = list()
        compatible_meta: Dict = None
        if self._img is None:
            raise RuntimeError("please call read() first then use get_data().")

        for img in self._img:
            header = self._get_meta_dict(img)
            header["original_affine"] = self._get_affine(img)
            header["affine"] = header["original_affine"].copy()
            header["spatial_shape"] = self._get_spatial_shape(img)
            img_array.append(self._get_array_data(img))

            if compatible_meta is None:
                compatible_meta = header
            else:
                if not np.allclose(header["affine"], compatible_meta["affine"]):
                    raise RuntimeError("affine matrix of all images should be same.")
                if not np.allclose(header["spatial_shape"], compatible_meta["spatial_shape"]):
                    raise RuntimeError("spatial_shape of all images should be same.")

        img_array_ = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array_, compatible_meta

    def _get_meta_dict(self, img: Image) -> Dict:
        """
        Get all the meta data of the image and convert to dict type.

        Args:
            img: a ITK image object loaded from a image file.

        """
        img_meta_dict = img.GetMetaDataDictionary()
        meta_dict = dict()
        for key in img_meta_dict.GetKeys():
            # ignore deprecated, legacy members that cause issues
            if key.startswith("ITK_original_"):
                continue
            meta_dict[key] = img_meta_dict[key]
        meta_dict["origin"] = np.asarray(img.GetOrigin())
        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        meta_dict["direction"] = itk.array_from_matrix(img.GetDirection())
        return meta_dict

    def _get_affine(self, img: Image) -> np.ndarray:
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.
        Construct Affine matrix based on direction, spacing, origin information.
        Refer to: https://github.com/RSIP-Vision/medio

        Args:
            img: a ITK image object loaded from a image file.

        """
        direction = itk.array_from_matrix(img.GetDirection())
        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())

        direction = np.asarray(direction)
        affine = np.eye(direction.shape[0] + 1)
        affine[(slice(-1), slice(-1))] = direction @ np.diag(spacing)
        affine[(slice(-1), -1)] = origin
        return affine

    def _get_spatial_shape(self, img: Image) -> Sequence:
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        Args:
            img: a ITK image object loaded from a image file.

        """
        return list(itk.size(img))

    def _get_array_data(self, img: Image) -> np.ndarray:
        """
        Get the raw array data of the image, converted to Numpy array.

        Args:
            img: a ITK image object loaded from a image file.

        """
        return itk.array_view_from_image(img, keep_axes=not self.c_order_axis_indexing)


class NibabelReader(ImageReader):
    """
    Load NIfTI format images based on Nibabel library.

    Args:
        as_closest_canonical: if True, load the image as closest to canonical axis format.

    """

    def __init__(self, as_closest_canonical: bool = False):
        super().__init__()
        self._img: Optional[Sequence[Nifti1Image]] = None
        self.as_closest_canonical = as_closest_canonical

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        Verify whether the specified file or files format is supported by Nibabel reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the subffixes.

        """
        suffixes: Sequence[str] = ["nii", "nii.gz"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[str], str, Nifti1Image], **kwargs):
        """
        Read image data from specified file or files, or set a Nibabel Image object.
        Note that the returned object is Nibabel image object or list of Nibabel image objects.
        `self._img` is always a list, even only has 1 image.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `nibabel.load` API. more details about available args:
                https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

        """
        self._img = list()
        if isinstance(data, Nifti1Image):
            self._img.append(data)
            return data

        filenames: Sequence[str] = ensure_tuple(data)
        for name in filenames:
            img = nib.load(name, **kwargs)
            img = correct_nifti_header_if_necessary(img)
            self._img.append(img)
        return self._img if len(filenames) > 1 else self._img[0]

    def get_data(self):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns 2 objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores in meta dict.
        If loading a list of files, stack them together and add a new dimension as first dimension,
        and use the meta data of the first image to represent the stacked result.

        """
        img_array: List[np.ndarray] = list()
        compatible_meta: Dict = None
        if self._img is None:
            raise RuntimeError("please call read() first then use get_data().")

        for img in self._img:
            header = self._get_meta_dict(img)
            header["original_affine"] = self._get_affine(img)
            header["affine"] = header["original_affine"].copy()
            if self.as_closest_canonical:
                img = nib.as_closest_canonical(img)
                header["affine"] = self._get_affine(img)
            header["as_closest_canonical"] = self.as_closest_canonical
            header["spatial_shape"] = self._get_spatial_shape(img)
            img_array.append(self._get_array_data(img))

            if compatible_meta is None:
                compatible_meta = header
            else:
                if not np.allclose(header["affine"], compatible_meta["affine"]):
                    raise RuntimeError("affine matrix of all images should be same.")
                if not np.allclose(header["spatial_shape"], compatible_meta["spatial_shape"]):
                    raise RuntimeError("spatial_shape of all images should be same.")

        img_array_ = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array_, compatible_meta

    def _get_meta_dict(self, img: Nifti1Image) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        return dict(img.header)

    def _get_affine(self, img: Nifti1Image) -> np.ndarray:
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        return img.affine

    def _get_spatial_shape(self, img: Nifti1Image) -> Sequence:
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        ndim = img.header["dim"][0]
        spatial_rank = min(ndim, 3)
        return list(img.header["dim"][1 : spatial_rank + 1])

    def _get_array_data(self, img: Nifti1Image) -> np.ndarray:
        """
        Get the raw array data of the image, converted to Numpy array.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        return np.asarray(img.dataobj)


class NumpyReader(ImageReader):
    """
    Load NPY or NPZ format data based on Numpy library, they can be arrays or pickled objects.
    A typical usage is to load the `mask` data for classification task.
    It can load part of the npz file with specified `npz_keys`.

    Args:
        npz_keys: if loading npz file, only load the specified keys, if None, load all the items.
            stack the loaded items together to construct a new first dimension.

    """

    def __init__(self, npz_keys: Optional[KeysCollection] = None):
        super().__init__()
        self._img: Optional[Sequence[Nifti1Image]] = None
        if npz_keys is not None:
            npz_keys = ensure_tuple(npz_keys)
        self.npz_keys = npz_keys

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        Verify whether the specified file or files format is supported by Numpy reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the subffixes.

        """
        suffixes: Sequence[str] = ["npz", "npy"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[str], str, np.ndarray], **kwargs):
        """
        Read image data from specified file or files, or set a Numpy array.
        Note that the returned object is Numpy array or list of Numpy arrays.
        `self._img` is always a list, even only has 1 image.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `numpy.load` API except `allow_pickle`. more details about available args:
                https://numpy.org/doc/stable/reference/generated/numpy.load.html

        """
        self._img = list()
        if isinstance(data, np.ndarray):
            self._img.append(data)
            return data

        filenames: Sequence[str] = ensure_tuple(data)
        for name in filenames:
            img = np.load(name, allow_pickle=True, **kwargs)
            if name.endswith(".npz"):
                # load expected items from NPZ file
                npz_keys = [f"arr_{i}" for i in range(len(img))] if self.npz_keys is None else self.npz_keys
                for k in npz_keys:
                    self._img.append(img[k])
            else:
                self._img.append(img)

        return self._img if len(filenames) > 1 else self._img[0]

    def get_data(self):
        """
        Extract data array and meta data from loaded data and return them.
        This function returns 2 objects, first is numpy array of image data, second is dict of meta data.
        It constructs `spatial_shape=data.shape` and stores in meta dict if the data is numpy array.
        If loading a list of files, stack them together and add a new dimension as first dimension,
        and use the meta data of the first image to represent the stacked result.

        """
        img_array: List[np.ndarray] = list()
        compatible_meta: Dict = None
        if self._img is None:
            raise RuntimeError("please call read() first then use get_data().")

        for img in self._img:
            header = dict()
            if isinstance(img, np.ndarray):
                header["spatial_shape"] = img.shape
            img_array.append(img)

            if compatible_meta is None:
                compatible_meta = header
            else:
                if not np.allclose(header["spatial_shape"], compatible_meta["spatial_shape"]):
                    raise RuntimeError("spatial_shape of all images should be same.")

        img_array_ = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array_, compatible_meta
