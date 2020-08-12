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
from typing import Any, Dict

import itk
import numpy as np


class ImageReader(ABC):
    """Abstract class to define interface APIs to load image files.

    """

    def __init__(self, img: Any = None):
        self._img = None

    @abstractmethod
    def read_image(self, filename: str):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_meta_data(self) -> Dict:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_array_data(self) -> np.ndarray:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def uncache(self):
        self._img = None


class ITKReader(ImageReader):
    def read_image(self, filename: str):
        self._img = itk.imread(filename)

    def get_meta_data(self) -> Dict:
        return self._img.GetMetaDataDictionary()

    def get_array_data(self) -> np.ndarray:
        return itk.array_from_image(self._img)
