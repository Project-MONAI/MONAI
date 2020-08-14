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
from typing import Any, List, Optional, Sequence, Union

import itk
import numpy as np


class ImageReader(ABC):
    """Abstract class to define interface APIs to load image files.

    """

    def __init__(self, suffixes: Optional[Union[str, Sequence[str]]] = None, img: Any = None):
        self.suffixes = suffixes
        self.img = img

    def verify_suffix(self, suffix: str):
        return False if self.suffixes is not None and suffix not in self.suffixes else True

    def uncache(self):
        self.img = None

    @abstractmethod
    def read_image(self, filename: str):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_meta_dict(self) -> List:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_affine(self) -> List:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_spatial_shape(self) -> List:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_array_data(self) -> np.ndarray:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class ITKReader(ImageReader):
    def read_image(self, filename: str):
        self.img = itk.imread(filename)

    def get_meta_dict(self) -> List:
        meta_dict = self.img.GetMetaDataDictionary()
        return {key: meta_dict[key] for key in meta_dict.GetKeys()}

    def get_affine(self) -> List:
        """
        Construct Affine matrix based on direction, spacing, origin information.
        Refer to: https://github.com/RSIP-Vision/medio

        """
        direction = itk.array_from_vnl_matrix(self.img.GetDirection().GetVnlMatrix().as_matrix())
        spacing = itk.array_from_vnl_vector(self.img.GetSpacing().GetVnlVector())
        origin = itk.array_from_vnl_vector(self.img.GetOrigin().GetVnlVector())

        direction = np.asarray(direction)
        affine = np.eye(direction.shape[0] + 1)
        affine[(slice(-1), slice(-1))] = direction @ np.diag(spacing)
        affine[(slice(-1), -1)] = origin
        return affine

    def get_spatial_shape(self) -> List:
        # don't support spatial dims greater than 3
        spatial_rank = min(self.img.GetImageDimension(), 3)
        meta_dict = self.img.GetMetaDataDictionary()
        spatial_shape = list()
        for i in range(1, spatial_rank + 1):
            spatial_shape.append(int(meta_dict[f"dim[{i}]"]))
        return spatial_shape

    def get_array_data(self) -> np.ndarray:
        return itk.array_from_image(self.img)
