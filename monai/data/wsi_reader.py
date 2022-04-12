# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import DtypeLike, PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import ensure_tuple, optional_import, require_pkg

CuImage, _ = optional_import("cucim", name="CuImage")

__all__ = ["BaseWSIReader", "WSIReader", "CuCIMWSIReader"]


class BaseWSIReader(ImageReader):
    """
    An abstract class defines APIs to load whole slide image files.

    Typical usage of an implementation of this class is:

    .. code-block:: python

        image_reader = MyWSIReader()
        wsi = image_reader.read(path_to_image)
        img_data, meta_data = image_reader.get_data(wsi)

    - The `read` call converts image filenames into image objects,
    - The `get_data` call fetches the image data, as well as meta data.
    - A reader should implement `verify_suffix` with the logic of checking the input filename
      by the filename extensions.

    """

    supported_formats: List[str] = []

    def __init__(self, level: int, **kwargs):
        super().__init__()
        self.level = level
        self.kwargs = kwargs
        self.metadata: Dict[Any, Any] = {}

    @property
    @abstractmethod
    def _reader(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def _get_size(self, wsi, level):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def _get_level_count(self, wsi):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def _get_patch(self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def _get_metadata(self, wsi, patch: np.ndarray, location: Tuple[int, int], size: Tuple[int, int], level: int):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read image data from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for backend reading API in `read()`, will override `self.kwargs` for existing keys.
                more details in `cuCIM`: https://github.com/rapidsai/cucim/blob/v21.12.00/cpp/include/cucim/cuimage.h#L100.

        Returns:
            image object or list of image objects

        """
        wsi_list: List = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = self._reader(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def get_data(
        self,
        wsi,
        location: Tuple[int, int] = (0, 0),
        size: Optional[Tuple[int, int]] = None,
        level: Optional[int] = None,
        dtype: DtypeLike = np.uint8,
    ):
        """
        Extract patchs as numpy array from WSI image and return them.

        Args:
            wsi: a whole slide image object loaded from a file
            location: (x_min, y_min) tuple giving the top left pixel in the level 0 reference frame,
            or list of tuples (default=(0, 0))
            size: (height, width) tuple giving the patch size, or list of tuples (default to full image size)
            This is the size of image at the given level (`level`)
            level: the level number, or list of level numbers (default=0)
            dtype: the data type of output image

        Returns:
            a tuples, where the first element is an image [CxHxW], and second element is a dictionary of metadata
        """
        # Verify magnification level
        if level is None:
            level = self.level
        max_level = self._get_level_count(wsi) - 1
        if level > max_level:
            raise ValueError(f"The maximum level of this image is {max_level} while level={level} is requested)!")

        # Verify location
        if location is None:
            location = (0, 0)
        wsi_size = self._get_size(wsi, level)
        if location[0] > wsi_size[0] or location[1] > wsi_size[1]:
            raise ValueError(f"Location is outside of the image: location={location}, image size={wsi_size}")

        # Verify size
        if size is None:
            if location != (0, 0):
                raise ValueError("Patch size should be defined to exctract patches.")
            size = self._get_size(wsi, level)
        else:
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"Patch size should be greater than zero, provided: patch size = {size}")

        # Extract a patch or the entire image
        patch = self._get_patch(wsi, location=location, size=size, level=level, dtype=dtype)

        # Verify patch image
        patch = self._verify_output(patch)

        # Set patch-related metadata
        metadata = self._get_metadata(wsi=wsi, patch=patch, location=location, size=size, level=level)

        return patch, metadata

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by WSI reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        return is_supported_format(filename, self.supported_formats)

    def _verify_output(self, patch: np.ndarray):
        """
        Verify image output
        """
        # check if the image has three dimensions (2D + color)
        if patch.ndim != 3:
            raise ValueError(
                f"The image dimension should be 3 but has {patch.ndim}. "
                "`WSIReader` is designed to work only with 2D colored images."
            )

        # check if the color channel is 3 (RGB) or 4 (RGBA)
        if patch.shape[0] not in [3, 4]:
            raise ValueError(
                f"The image should have three or four color channels but has {patch.shape[0]}. "
                "`WSIReader` is designed to work only with 2D colored images."
            )

        # remove alpha channel if exist (RGBA)
        if patch.shape[0] > 3:
            patch = patch[:3]

        return patch


class WSIReader(BaseWSIReader):
    def __init__(self, backend="cucim", level: int = 0, **kwargs):
        super().__init__(level, **kwargs)
        self.backend = backend.lower()
        if self.backend == "cucim":
            self.backend_lib = CuCIMWSIReader(level=level, **kwargs)
        else:
            raise ValueError("The supported backends are: cucim")
        self.supported_formats = self.backend_lib.supported_formats

    @property
    def _reader(self):
        return self.backend_lib._reader

    def _get_level_count(self, wsi):
        return self.backend_lib._get_level_count(wsi)

    def _get_size(self, wsi, level):
        return self.backend_lib._get_size(wsi, level)

    def _get_metadata(self, wsi, patch: np.ndarray, location: Tuple[int, int], size: Tuple[int, int], level: int):
        return self.backend_lib._get_metadata(wsi=wsi, patch=patch, size=size, location=location, level=level)

    def _get_patch(self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike):
        return self.backend_lib._get_patch(wsi=wsi, location=location, size=size, level=level, dtype=dtype)


@require_pkg(pkg_name="cucim")
class CuCIMWSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches without loading the whole slide image into the memory.

    Args:
        level: the whole slide image level at which the image is extracted. (default=0)
            This is overridden if the level argument is provided in `get_data`.
        kwargs: additional args for backend reading API in `read()`, more details in `cuCIM`:
            https://github.com/rapidsai/cucim/blob/v21.12.00/cpp/include/cucim/cuimage.h#L100.

    """

    supported_formats = ["tif", "tiff", "svs"]

    def __init__(self, level: int = 0, **kwargs):
        super().__init__(level, **kwargs)

    @property
    def _reader(self):
        return CuImage

    @staticmethod
    def _get_level_count(wsi):
        return wsi.resolutions["level_count"]

    @staticmethod
    def _get_size(wsi, level):
        return wsi.resolutions["level_dimensions"][level][::-1]

    def _get_metadata(self, wsi, patch: np.ndarray, location: Tuple[int, int], size: Tuple[int, int], level: int):
        metadata: Dict = {
            "backend": "cucim",
            "spatial_shape": np.asarray(patch.shape[1:]),
            "original_channel_dim": -1,
            "location": location,
            "size": size,
            "level": level,
        }
        return metadata

    def _get_patch(self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike):
        """
        Extract a patch based on given output from the given whole slide image
        Args:

        Returns:
            a numpy array with dimesion of [3xWxH]
        """
        # extract a patch (or the entire image)
        # reverse the order of location and size to become WxH for cuCIM
        patch = wsi.read_region(location=location[::-1], size=size[::-1], level=level)

        # convert to numpy
        patch = np.asarray(patch, dtype=dtype)

        # make it channel first
        patch = EnsureChannelFirst()(patch, {"original_channel_dim": -1})

        return patch
