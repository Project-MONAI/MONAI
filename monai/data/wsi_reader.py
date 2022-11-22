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
from os.path import abspath
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config import DtypeLike, PathLike
from monai.data.image_reader import ImageReader, _stack_images
from monai.data.utils import is_supported_format
from monai.utils import WSIPatchKeys, ensure_tuple, optional_import, require_pkg

CuImage, _ = optional_import("cucim", name="CuImage")
OpenSlide, _ = optional_import("openslide", name="OpenSlide")
TiffFile, _ = optional_import("tifffile", name="TiffFile")

__all__ = ["BaseWSIReader", "WSIReader", "CuCIMWSIReader", "OpenSlideWSIReader", "TiffFileWSIReader"]


class BaseWSIReader(ImageReader):
    """
    An abstract class that defines APIs to load patches from whole slide image files.

    Typical usage of a concrete implementation of this class is:

    .. code-block:: python

        image_reader = MyWSIReader()
        wsi = image_reader.read(, **kwargs)
        img_data, meta_data = image_reader.get_data(wsi)

    - The `read` call converts an image filename into whole slide image object,
    - The `get_data` call fetches the image data, as well as metadata.

    The following methods needs to be implemented for any concrete implementation of this class:

    - `read` reads a whole slide image object from a given file
    - `get_size` returns the size of the whole slide image of a given wsi object at a given level.
    - `get_level_count` returns the number of levels in the whole slide image
    - `_get_patch` extracts and returns a patch image form the whole slide image
    - `_get_metadata` extracts and returns metadata for a whole slide image and a specific patch.


    """

    supported_suffixes: List[str] = []
    backend = ""

    def __init__(self, level: int = 0, channel_dim: int = 0, **kwargs):
        super().__init__()
        self.level = level
        self.channel_dim = channel_dim
        self.kwargs = kwargs
        self.metadata: Dict[Any, Any] = {}

    @abstractmethod
    def get_size(self, wsi, level: Optional[int] = None) -> Tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_level_count(self, wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_downsample_ratio(self, wsi, level: Optional[int] = None) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_file_path(self, wsi) -> str:
        """Return the file path for the WSI object"""
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_mpp(self, wsi, level: Optional[int] = None) -> Tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def _get_patch(
        self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _get_metadata(
        self, wsi, patch: np.ndarray, location: Tuple[int, int], size: Tuple[int, int], level: int
    ) -> Dict:
        """
        Returns metadata of the extracted patch from the whole slide image.

        Args:
            wsi: the whole slide image object, from which the patch is loaded
            patch: extracted patch from whole slide image
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0

        """
        if self.channel_dim >= len(patch.shape) or self.channel_dim < -len(patch.shape):
            raise ValueError(
                f"The desired channel_dim ({self.channel_dim}) is out of bound for image shape: {patch.shape}"
            )
        channel_dim: int = self.channel_dim + (len(patch.shape) if self.channel_dim < 0 else 0)
        metadata: Dict = {
            "backend": self.backend,
            "original_channel_dim": channel_dim,
            "spatial_shape": np.array(patch.shape[:channel_dim] + patch.shape[channel_dim + 1 :]),
            WSIPatchKeys.COUNT.value: 1,
            WSIPatchKeys.PATH.value: self.get_file_path(wsi),
            WSIPatchKeys.LOCATION.value: np.asarray(location),
            WSIPatchKeys.SIZE.value: np.asarray(size),
            WSIPatchKeys.LEVEL.value: level,
        }
        return metadata

    def get_data(
        self,
        wsi,
        location: Tuple[int, int] = (0, 0),
        size: Optional[Tuple[int, int]] = None,
        level: Optional[int] = None,
        dtype: DtypeLike = np.uint8,
        mode: str = "RGB",
    ) -> Tuple[np.ndarray, Dict]:
        """
        Verifies inputs, extracts patches from WSI image and generates metadata, and return them.

        Args:
            wsi: a whole slide image object loaded from a file or a list of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'

        Returns:
            a tuples, where the first element is an image patch [CxHxW] or stack of patches,
                and second element is a dictionary of metadata
        """
        patch_list: List = []
        metadata_list: List = []
        # CuImage object is iterable, so ensure_tuple won't work on single object
        if not isinstance(wsi, List):
            wsi = [wsi]
        for each_wsi in ensure_tuple(wsi):
            # Verify magnification level
            if level is None:
                level = self.level
            max_level = self.get_level_count(each_wsi) - 1
            if level > max_level:
                raise ValueError(f"The maximum level of this image is {max_level} while level={level} is requested)!")

            # Verify location
            if location is None:
                location = (0, 0)
            wsi_size = self.get_size(each_wsi, 0)
            if location[0] > wsi_size[0] or location[1] > wsi_size[1]:
                raise ValueError(f"Location is outside of the image: location={location}, image size={wsi_size}")

            # Verify size
            if size is None:
                if location != (0, 0):
                    raise ValueError("Patch size should be defined to extract patches.")
                size = self.get_size(each_wsi, level)
            else:
                if size[0] <= 0 or size[1] <= 0:
                    raise ValueError(f"Patch size should be greater than zero, provided: patch size = {size}")

            # Extract a patch or the entire image
            patch = self._get_patch(each_wsi, location=location, size=size, level=level, dtype=dtype, mode=mode)

            # check if the image has three dimensions (2D + color)
            if patch.ndim != 3:
                raise ValueError(
                    f"The image dimension should be 3 but has {patch.ndim}. "
                    "`WSIReader` is designed to work only with 2D images with color channel."
                )
            # Check if there are four color channels for RGBA
            if mode == "RGBA":
                if patch.shape[self.channel_dim] != 4:
                    raise ValueError(
                        f"The image is expected to have four color channels in '{mode}' mode but has "
                        f"{patch.shape[self.channel_dim]}."
                    )
            # Check if there are three color channels for RGB
            elif mode in "RGB" and patch.shape[self.channel_dim] != 3:
                raise ValueError(
                    f"The image is expected to have three color channels in '{mode}' mode but has "
                    f"{patch.shape[self.channel_dim]}. "
                )
            # Get patch-related metadata
            metadata: dict = self._get_metadata(wsi=each_wsi, patch=patch, location=location, size=size, level=level)
            # Create a list of patches and metadata
            patch_list.append(patch)
            metadata_list.append(metadata)
        if len(wsi) > 1:
            if len({m["original_channel_dim"] for m in metadata_list}) > 1:
                raise ValueError("original_channel_dim is not consistent across wsi objects.")
            if len({tuple(m["spatial_shape"]) for m in metadata_list}) > 1:
                raise ValueError("spatial_shape is not consistent across wsi objects.")
            for key in WSIPatchKeys:
                metadata[key] = [m[key] for m in metadata_list]
        return _stack_images(patch_list, metadata), metadata

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by WSI reader.

        The list of supported suffixes are read from `self.supported_suffixes`.

        Args:
            filename: filename or a list of filenames to read.

        """
        return is_supported_format(filename, self.supported_suffixes)


class WSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using different backend libraries

    Args:
        backend: the name of backend whole slide image reader library, the default is cuCIM.
        level: the level at which patches are extracted.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        num_workers: number of workers for multi-thread image loading (cucim backend only).
        kwargs: additional arguments to be passed to the backend library

    """

    def __init__(self, backend="cucim", level: int = 0, channel_dim: int = 0, **kwargs):
        super().__init__(level, channel_dim, **kwargs)
        self.backend = backend.lower()
        self.reader: Union[CuCIMWSIReader, OpenSlideWSIReader, TiffFileWSIReader]
        if self.backend == "cucim":
            self.reader = CuCIMWSIReader(level=level, channel_dim=channel_dim, **kwargs)
        elif self.backend == "openslide":
            self.reader = OpenSlideWSIReader(level=level, channel_dim=channel_dim, **kwargs)
        elif self.backend == "tifffile":
            self.reader = TiffFileWSIReader(level=level, channel_dim=channel_dim, **kwargs)
        else:
            raise ValueError(
                f"The supported backends are cucim, openslide, and tifffile but '{self.backend}' was given."
            )
        self.supported_suffixes = self.reader.supported_suffixes

    def get_level_count(self, wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return self.reader.get_level_count(wsi)

    def get_size(self, wsi, level: Optional[int] = None) -> Tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return self.reader.get_size(wsi, level)

    def get_downsample_ratio(self, wsi, level: Optional[int] = None) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return self.reader.get_downsample_ratio(wsi, level)

    def get_file_path(self, wsi) -> str:
        """Return the file path for the WSI object"""
        return self.reader.get_file_path(wsi)

    def get_mpp(self, wsi, level: Optional[int] = None) -> Tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return self.reader.get_mpp(wsi, level)

    def _get_patch(
        self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'

        """
        return self.reader._get_patch(wsi=wsi, location=location, size=size, level=level, dtype=dtype, mode=mode)

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for the reader module (overrides `self.kwargs` for existing keys).

        Returns:
            whole slide image object or list of such objects

        """
        return self.reader.read(data=data, **kwargs)


@require_pkg(pkg_name="cucim")
class CuCIMWSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using cuCIM library.

    Args:
        level: the whole slide image level at which the image is extracted. (default=0)
            This is overridden if the level argument is provided in `get_data`.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        num_workers: number of workers for multi-thread image loading
        kwargs: additional args for `cucim.CuImage` module:
            https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "cucim"

    def __init__(self, level: int = 0, channel_dim: int = 0, num_workers: int = 0, **kwargs):
        super().__init__(level, channel_dim, **kwargs)
        self.num_workers = num_workers

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return wsi.resolutions["level_count"]  # type: ignore

    def get_size(self, wsi, level: Optional[int] = None) -> Tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return (wsi.resolutions["level_dimensions"][level][1], wsi.resolutions["level_dimensions"][level][0])

    def get_downsample_ratio(self, wsi, level: Optional[int] = None) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return wsi.resolutions["level_downsamples"][level]  # type: ignore

    @staticmethod
    def get_file_path(wsi) -> str:
        """Return the file path for the WSI object"""
        return str(abspath(wsi.path))

    def get_mpp(self, wsi, level: Optional[int] = None) -> Tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        factor = float(wsi.resolutions["level_downsamples"][level])
        return (wsi.metadata["cucim"]["spacing"][1] * factor, wsi.metadata["cucim"]["spacing"][0] * factor)

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.
                For more details look at https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h

        Returns:
            whole slide image object or list of such objects

        """
        wsi_list: List = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = CuImage(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'

        """
        # Extract a patch or the entire image
        # (reverse the order of location and size to become WxH for cuCIM)
        patch: np.ndarray = wsi.read_region(
            location=location[::-1], size=size[::-1], level=level, num_workers=self.num_workers
        )

        # Convert to numpy
        patch = np.asarray(patch, dtype=dtype)

        # Make the channel to desired dimensions
        patch = np.moveaxis(patch, -1, self.channel_dim)

        # Check if the color channel is 3 (RGB) or 4 (RGBA)
        if mode in "RGB":
            if patch.shape[self.channel_dim] not in [3, 4]:
                raise ValueError(
                    f"The image is expected to have three or four color channels in '{mode}' mode but has "
                    f"{patch.shape[self.channel_dim]}. "
                )
            patch = patch[:3]

        return patch


@require_pkg(pkg_name="openslide")
class OpenSlideWSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using OpenSlide library.

    Args:
        level: the whole slide image level at which the image is extracted. (default=0)
            This is overridden if the level argument is provided in `get_data`.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        kwargs: additional args for `openslide.OpenSlide` module.

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "openslide"

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return wsi.level_count  # type: ignore

    def get_size(self, wsi, level: Optional[int] = None) -> Tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return (wsi.level_dimensions[level][1], wsi.level_dimensions[level][0])

    def get_downsample_ratio(self, wsi, level: Optional[int] = None) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return wsi.level_downsamples[level]  # type: ignore

    @staticmethod
    def get_file_path(wsi) -> str:
        """Return the file path for the WSI object"""
        return str(abspath(wsi._filename))

    def get_mpp(self, wsi, level: Optional[int] = None) -> Tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level
        unit = wsi.properties["tiff.ResolutionUnit"]
        if unit == "centimeter":
            factor = 10000.0
        elif unit == "milimeter":
            factor = 1000.0
        elif unit == "micrometer":
            factor = 1.0
        elif unit == "inch":
            factor = 25400.0
        else:
            raise ValueError(f"The resolution unit is not a valid tiff resolution: {unit}")

        factor *= wsi.level_downsamples[level]
        return (factor / float(wsi.properties["tiff.YResolution"]), factor / float(wsi.properties["tiff.XResolution"]))

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.

        Returns:
            whole slide image object or list of such objects

        """
        wsi_list: List = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = OpenSlide(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'

        """
        # Extract a patch or the entire image
        # (reverse the order of location and size to become WxH for OpenSlide)
        pil_patch = wsi.read_region(location=location[::-1], size=size[::-1], level=level)

        # convert to RGB/RGBA
        pil_patch = pil_patch.convert(mode)

        # Convert to numpy
        patch = np.asarray(pil_patch, dtype=dtype)

        # Make the channel to desired dimensions
        patch = np.moveaxis(patch, -1, self.channel_dim)

        return patch


@require_pkg(pkg_name="tifffile")
class TiffFileWSIReader(BaseWSIReader):
    """
    Read whole slide images and extract patches using TiffFile library.

    Args:
        level: the whole slide image level at which the image is extracted. (default=0)
            This is overridden if the level argument is provided in `get_data`.
        channel_dim: the desired dimension for color channel. Default to 0 (channel first).
        kwargs: additional args for `tifffile.TiffFile` module.

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "tifffile"

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return len(wsi.pages)

    def get_size(self, wsi, level: Optional[int] = None) -> Tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return (wsi.pages[level].imagelength, wsi.pages[level].imagewidth)

    def get_downsample_ratio(self, wsi, level: Optional[int] = None) -> float:
        """
        Returns the down-sampling ratio of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        return float(wsi.pages[0].imagelength) / float(wsi.pages[level].imagelength)

    @staticmethod
    def get_file_path(wsi) -> str:
        """Return the file path for the WSI object"""
        return str(abspath(wsi.filehandle.path))

    def get_mpp(self, wsi, level: Optional[int] = None) -> Tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        unit = wsi.pages[level].tags["ResolutionUnit"].value
        if unit == unit.CENTIMETER:
            factor = 10000.0
        elif unit == unit.MILLIMETER:
            factor = 1000.0
        elif unit == unit.MICROMETER:
            factor = 1.0
        elif unit == unit.INCH:
            factor = 25400.0
        else:
            raise ValueError(f"The resolution unit is not a valid tiff resolution or missing: {unit.name}")

        # Here x and y resolutions are rational numbers so each of them is represented by a tuple.
        yres = wsi.pages[level].tags["YResolution"].value
        xres = wsi.pages[level].tags["XResolution"].value
        return (factor * yres[1] / yres[0], factor * xres[1] / xres[0])

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.

        Returns:
            whole slide image object or list of such objects

        """
        wsi_list: List = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = TiffFile(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: Tuple[int, int], size: Tuple[int, int], level: int, dtype: DtypeLike, mode: str
    ) -> np.ndarray:
        """
        Extracts and returns a patch image form the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file or a lis of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'

        """
        # Load the entire image
        wsi_image: np.ndarray = wsi.asarray(level=level).astype(dtype)
        if len(wsi_image.shape) < 3:
            wsi_image = wsi_image[..., None]

        # Extract patch
        downsampling_ratio = self.get_downsample_ratio(wsi=wsi, level=level)
        location_ = [round(location[i] / downsampling_ratio) for i in range(len(location))]
        patch = wsi_image[location_[0] : location_[0] + size[0], location_[1] : location_[1] + size[1], :].copy()

        # Make the channel to desired dimensions
        patch = np.moveaxis(patch, -1, self.channel_dim)

        # Check if the color channel is 3 (RGB) or 4 (RGBA)
        if mode in "RGB":
            if patch.shape[self.channel_dim] not in [3, 4]:
                raise ValueError(
                    f"The image is expected to have three or four color channels in '{mode}' mode but has "
                    f"{patch.shape[self.channel_dim]}. "
                )
            patch = patch[:3]

        return patch
