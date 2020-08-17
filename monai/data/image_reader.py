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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from monai.data.utils import correct_nifti_header_if_necessary
from monai.utils import optional_import

itk, _ = optional_import("itk", allow_namespace_pkg=True)
nib, _ = optional_import("nibabel")


class ImageReader(ABC):
    """Abstract class to define interface APIs to load image files.
    users need to call `read_image` to load image and then use other APIs
    to get image data or properties from meta data.

    Args:
        img: image to initialize the reader, this is for the usage that the image data
            is already in memory and no need to read from file again, default is None.
        as_closest_canonical: if True, load the image as closest to canonical axis format.

    """

    def __init__(self, img: Optional[Any] = None, as_closest_canonical: bool = False):
        self.img = img
        self.as_closest_canonical = as_closest_canonical
        self._suffixes: Optional[Sequence] = None

    def get_suffixes(self):
        """
        Get the supported image file suffixes of current reader.
        Default is None, support all kinds of image format.

        """
        return self._suffixes

    def verify_suffix(self, suffix: str):
        """
        Verify whether the specified file matches supported suffixes.
        If supported suffixes is None, skip the verification.

        """
        return False if self._suffixes is not None and suffix not in self._suffixes else True

    def uncache(self):
        """
        Release image object and other cache data.

        """
        self.img = None

    @abstractmethod
    def convert(self):
        """
        Convert the image if necessary.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def read_image(self, filename: str):
        """
        Read image data from specified file.
        Note that different readers return different image data type.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_meta_dict(self) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_affine(self) -> np.ndarray:
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_spatial_shape(self) -> List:
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_array_data(self) -> np.ndarray:
        """
        Get the raw array data of the image, converted to Numpy array.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ITKReader(ImageReader):
    """
    Load medical images based on ITK library.
    All the supported image formats can be found:
    https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/IO

    """

    def convert(self, as_closest_canonical: Optional[bool] = None):
        """
        Convert the image as closest to canonical axis format.

        """
        # FIXME: need to add support later
        pass

    def read_image(self, filename: str):
        """
        Read image data from specified file.
        Note that the returned object is ITK image object.

        """
        self.img = itk.imread(filename)

    def get_meta_dict(self) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.

        """
        img_meta_dict = self.img.GetMetaDataDictionary()
        meta_dict = dict()
        for key in img_meta_dict.GetKeys():
            # ignore deprecated, legacy members that cause issues
            if key.startswith("ITK_original_"):
                continue
            meta_dict[key] = img_meta_dict[key]
        meta_dict["origin"] = np.asarray(self.img.GetOrigin())
        meta_dict["spacing"] = np.asarray(self.img.GetSpacing())
        meta_dict["direction"] = itk.array_from_matrix(self.img.GetDirection())
        return meta_dict

    def get_affine(self) -> np.ndarray:
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.
        Construct Affine matrix based on direction, spacing, origin information.
        Refer to: https://github.com/RSIP-Vision/medio

        """
        direction = itk.array_from_matrix(self.img.GetDirection())
        spacing = np.asarray(self.img.GetSpacing())
        origin = np.asarray(self.img.GetOrigin())

        direction = np.asarray(direction)
        affine = np.eye(direction.shape[0] + 1)
        affine[(slice(-1), slice(-1))] = direction @ np.diag(spacing)
        affine[(slice(-1), -1)] = origin
        return affine

    def get_spatial_shape(self) -> List:
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        """
        return list(itk.size(self.img))

    def get_array_data(self) -> np.ndarray:
        """
        Get the raw array data of the image, converted to Numpy array.

        """
        return itk.array_view_from_image(self.img, keep_axes=True)


class NibabelReader(ImageReader):
    """
    Load NIfTI format images based on Nibabel library.

    Args:
        img: image to initialize the reader, this is for the usage that the image data
            is already in memory and no need to read from file again, default is None.
        as_closest_canonical: if True, load the image as closest to canonical axis format.

    """

    def __init__(self, img: Optional[Any] = None, as_closest_canonical: bool = False):
        super().__init__(img, as_closest_canonical)
        self._suffixes: [Sequence] = ["nii", "nii.gz"]

    def convert(self, as_closest_canonical: Optional[bool] = None):
        """
        Convert the image as closest to canonical axis format.

        """
        if as_closest_canonical is None:
            as_closest_canonical = self.as_closest_canonical
        if as_closest_canonical:
            self.img = nib.as_closest_canonical(self.img)

    def read_image(self, filename: str):
        """
        Read image data from specified file.
        Note that the returned object is Nibabel image object.

        """
        img = nib.load(filename)
        img = correct_nifti_header_if_necessary(img)
        if self.as_closest_canonical:
            img = nib.as_closest_canonical(img)
        self.img = img

    def get_meta_dict(self) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.

        """
        meta_data = dict(self.img.header)
        meta_data["as_closest_canonical"] = self.as_closest_canonical
        return meta_data

    def get_affine(self) -> np.ndarray:
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        """
        return self.img.affine

    def get_spatial_shape(self) -> List:
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        """
        ndim = self.img.header["dim"][0]
        spatial_rank = min(ndim, 3)
        return self.img.header["dim"][1 : spatial_rank + 1]

    def get_array_data(self) -> np.ndarray:
        """
        Get the raw array data of the image, converted to Numpy array.

        """
        return np.array(self.img.get_fdata())

    def uncache(self):
        """
        Release image object and other cache data.

        """
        if self.img is not None:
            self.img.uncache()
            super().uncache()
