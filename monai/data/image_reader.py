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
from typing import Any, Dict, Tuple, Optional, Sequence, Union

import numpy as np

from monai.data.utils import correct_nifti_header_if_necessary
from monai.utils import optional_import, ensure_tuple

itk, _ = optional_import("itk", allow_namespace_pkg=True)
nib, _ = optional_import("nibabel")


class ImageReader(ABC):
    """Abstract class to define interface APIs to load image files.
    users need to call `read` to load image and then use `get_data`
    to get the image data and properties from meta data.

    Args:
        img: image to initialize the reader, this is for the usage that the image data
            is already in memory and no need to read from file again, default is None.

    """

    def __init__(self, img: Optional[Any] = None) -> None:
        self.img = img
        self._suffixes: Optional[Sequence[str]] = None

    def get_suffixes(self) -> Sequence[str]:
        """
        Get the supported image file suffixes of current reader.
        Default is None, support all kinds of image format.

        """
        return self._suffixes

    def verify_suffix(self, filename: str) -> bool:
        """
        Verify whether the specified file or files match supported suffixes.
        If supported suffixes is None, skip the verification and return True.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the subffixes.
        """

        if self._suffixes is None:
            return True

        supported_format: bool = True
        filenames: Sequence[str] = ensure_tuple(filename)
        for name in filenames:
            suffixes: Sequence[str] = name.split(".")
            passed: bool = False
            for i in range(len(suffixes) - 1):
                if ".".join(suffixes[-(i + 2) : -1]) in self._suffixes:
                    passed = True
                    break
            if not passed:
                supported_format = False
                break

        return supported_format

    def uncache(self) -> None:
        """
        Release image object and other cache data.

        """
        self.img = None

    @abstractmethod
    def read(self, filename: str) -> Union[Sequence[Any], Any]:
        """
        Read image data from specified file or files and save to `self.img`.
        Note that it returns the raw data, so different readers return different image data type.

        Args:
            filename: file name or a list of file names to read.

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
        img: image to initialize the reader, this is for the usage that the image data
            is already in memory and no need to read from file again, default is None.
        keep_axes: default to `True`. if `False`, the numpy array will have C-order indexing.
            this is the reverse of how indices are specified in ITK, i.e. k,j,i versus i,j,k.
            however C-order indexing is expected by most algorithms in numpy / scipy.

    """
    def __init__(self, img: Optional[itk.Image] = None, keep_axes: bool = True):
        super().__init__(img=img)
        self.keep_axes = keep_axes

    def read(self, filename: str):
        """
        Read image data from specified file or files.
        Note that the returned object is ITK image object or list of ITK image objects.
        `self.img` is always a list, even only has 1 image.

        Args:
            filename: file name or a list of file names to read.

        """
        filenames: Sequence[str] = ensure_tuple(filename)
        self.img: Sequence[itk.Image] = list()
        for name in filenames:
            self.img.append(itk.imread(name))
        return self.img if len(filenames) > 1 else self.img[0]

    def get_data(self):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns 2 objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores in meta dict.
        If loading a list of files, stack them together and add a new dimension as first dimension,
        and use the meta data of the first image to represent the stacked result.

        """
        img_array: Sequence[np.ndarray] = list()
        compatible_meta: Dict = None
        for img in self.img:
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

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array, compatible_meta

    def _get_meta_dict(self, img: itk.Image) -> Dict:
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

    def _get_affine(self, img: itk.Image) -> np.ndarray:
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

    def _get_spatial_shape(self, img: itk.Image) -> Sequence:
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        Args:
            img: a ITK image object loaded from a image file.

        """
        return list(itk.size(img))

    def _get_array_data(self, img: itk.Image) -> np.ndarray:
        """
        Get the raw array data of the image, converted to Numpy array.

        Args:
            img: a ITK image object loaded from a image file.

        """
        return itk.array_view_from_image(img, keep_axes=self.keep_axes)


class NibabelReader(ImageReader):
    """
    Load NIfTI format images based on Nibabel library.

    Args:
        img: image to initialize the reader, this is for the usage that the image data
            is already in memory and no need to read from file again, default is None.
        as_closest_canonical: if True, load the image as closest to canonical axis format.

    """

    def __init__(self, img: Optional[Any] = None, as_closest_canonical: bool = False):
        super().__init__(img)
        self._suffixes: [Sequence[str]] = ["nii", "nii.gz"]
        self.as_closest_canonical = as_closest_canonical

    def read(self, filename: str):
        """
        Read image data from specified file or files.
        Note that the returned object is Nibabel image object or list of Nibabel image objects.
        `self.img` is always a list, even only has 1 image.

        Args:
            filename: file name or a list of file names to read.

        """
        filenames: Sequence[str] = ensure_tuple(filename)
        self.img: Sequence[nib.nifti1.Nifti1Image] = list()
        for name in filenames:
            img = nib.load(name)
            img = correct_nifti_header_if_necessary(img)
            self.img.append(img)
        return self.img if len(filenames) > 1 else self.img[0]

    def get_data(self):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns 2 objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores in meta dict.
        If loading a list of files, stack them together and add a new dimension as first dimension,
        and use the meta data of the first image to represent the stacked result.

        """
        img_array: Sequence[np.ndarray] = list()
        compatible_meta: Dict = None
        for img in self.img:
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

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array, compatible_meta

    def _get_meta_dict(self, img: nib.nifti1.Nifti1Image) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        return dict(img.header)

    def _get_affine(self, img: nib.nifti1.Nifti1Image) -> np.ndarray:
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        return img.affine

    def _get_spatial_shape(self, img: nib.nifti1.Nifti1Image) -> Sequence:
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        ndim = img.header["dim"][0]
        spatial_rank = min(ndim, 3)
        return img.header["dim"][1 : spatial_rank + 1]

    def _get_array_data(self, img: nib.nifti1.Nifti1Image) -> np.ndarray:
        """
        Get the raw array data of the image, converted to Numpy array.

        Args:
            img: a Nibabel image object loaded from a image file.

        """
        return np.array(img.get_fdata())

    def uncache(self):
        """
        Release image object and other cache data.

        """
        if self.img is not None:
            for img in self.img:
                img.uncache()
        super().uncache()
