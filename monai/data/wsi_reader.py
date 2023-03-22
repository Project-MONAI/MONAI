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

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from os.path import abspath
from typing import Any

import numpy as np
import torch

from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data.image_reader import ImageReader, _stack_images
from monai.data.utils import is_supported_format
from monai.utils import (
    WSIPatchKeys,
    dtype_numpy_to_torch,
    dtype_torch_to_numpy,
    ensure_tuple,
    ensure_tuple_rep,
    optional_import,
    require_pkg,
)

OpenSlide, _ = optional_import("openslide", name="OpenSlide")
TiffFile, _ = optional_import("tifffile", name="TiffFile")

__all__ = ["BaseWSIReader", "WSIReader", "CuCIMWSIReader", "OpenSlideWSIReader", "TiffFileWSIReader"]


class BaseWSIReader(ImageReader):
    """
    An abstract class that defines APIs to load patches from whole slide image files.

    Args:
        level: the whole slide image level at which the image is extracted.
        channel_dim: the desired dimension for color channel.
        dtype: the data type of output image.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, e.g., "RGB" or "RGBA".
        mpp:
        power:
        mpp_rtol:
        mpp_atol:
        power_rtol:
        power_atol:
        kwargs: additional args for the reader

    Typical usage of a concrete implementation of this class is:

    .. code-block:: python

        image_reader = MyWSIReader()
        wsi = image_reader.read(filepath, **kwargs)
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

    supported_suffixes: list[str] = []
    backend = ""

    def __init__(
        self,
        level: int | None,
        mpp: float | tuple[float, float] | None,
        power: int | None,
        channel_dim: int,
        dtype: DtypeLike | torch.dtype,
        device: torch.device | str | None,
        mode: str,
        mpp_rtol: float,
        mpp_atol: float,
        power_rtol: float,
        power_atol: float,
        **kwargs,
    ):
        super().__init__()
        self.level = level
        self.channel_dim = channel_dim
        self.set_dtype(dtype)
        self.set_device(device)
        self.mode = mode
        self.kwargs = kwargs
        self.mpp = mpp if mpp is None else ensure_tuple_rep(mpp, 2)
        self.power = power
        self.mpp_rtol = mpp_rtol
        self.mpp_atol = mpp_atol
        self.power_rtol = power_rtol
        self.power_atol = power_atol
        self.metadata: dict[Any, Any] = {}

    def set_dtype(self, dtype):
        self.dtype: torch.dtype | np.dtype
        if isinstance(dtype, torch.dtype):
            self.dtype = dtype
        else:
            self.dtype = np.dtype(dtype)

    def set_device(self, device):
        if device is None or isinstance(device, (torch.device, str)):
            self.device = device
        else:
            raise ValueError(f"`device` must be `torch.device`, `str` or `None` but {type(device)} is given.")

    @abstractmethod
    def get_size(self, wsi, level: int | None = None) -> tuple[int, int]:
        """
        Returns the size (height, width) of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the size is calculated

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _get_valid_level(
        self,
        wsi,
        level: int | None,
        mpp: tuple[float, float] | None,
        power: int | None,
    ) -> int:
        """
        Returns the level associated to the resolution parameter in the whole slide image.
        """

        if mpp is not None:
            mpp = ensure_tuple_rep(mpp, 2)

        # Try instance parameters if no resolution is provided
        if mpp is None and power is None and level is None:
            mpp = self.mpp
            power = self.power
            level = self.level

        resolution = [val[0] for val in [("level", level), ("mpp", mpp), ("power", power)] if val[1] is not None]
        # Check if only one resolution parameter is provided
        if len(resolution) > 1:
            raise ValueError(f"Only one of `level`, `mpp`, or `power` should be provided. {resolution} are provided.")
        # Set the default value if no resolution parameter is provided.
        if len(resolution) < 1:
            level = 0

        n_levels = self.get_level_count(wsi)

        if level is not None:
            if level >= n_levels:
                raise ValueError(f"The maximum level of this image is {n_levels-1} while level={level} is requested)!")

        elif mpp is not None:
            if self.get_mpp(wsi, 0) is None:
                raise ValueError(
                    "mpp is not defined in this whole slide image, please use `level` (or `power`) instead."
                )
            available_mpps = [self.get_mpp(wsi, level) for level in range(n_levels)]
            if mpp in available_mpps:
                valid_mpp = mpp
            else:
                valid_mpp = min(available_mpps, key=lambda x: abs(x[0] - mpp[0]) + abs(x[1] - mpp[1]))
                for i in range(2):
                    if abs(valid_mpp[i] - mpp[i]) > self.mpp_atol + self.mpp_rtol * abs(mpp[i]):
                        raise ValueError(
                            f"The requested mpp {mpp} does not exist in this whole slide image"
                            f"(with mpp_rtol={self.mpp_rtol} and mpp_atol={self.mpp_atol}). "
                            f"Here is the list of available mpps: {available_mpps}. "
                            f"The closest matching available mpp is {valid_mpp}."
                            "Please consider changing the tolerances or use another mpp."
                        )
            level = available_mpps.index(valid_mpp)

        elif power is not None:
            available_powers = [self.get_power(wsi, level) for level in range(n_levels)]
            if power in available_powers:
                valid_power = power
            else:
                valid_power = min(available_powers, key=lambda x: abs(x - power))  # type: ignore
                if abs(valid_power - power) > self.power_atol + self.power_rtol * abs(power):
                    raise ValueError(
                        f"The requested power ({power}) does not exist in this whole slide image"
                        f"(with power_rtol={self.power_rtol} and power_atol={self.power_atol})."
                        f" The closest matching available power is {valid_power}."
                        "Please consider changing the tolerances or use another power."
                    )
            level = available_powers.index(valid_power)

        return level

    @abstractmethod
    def get_level_count(self, wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_downsample_ratio(self, wsi, level: int | None = None) -> float:
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
    def get_mpp(self, wsi, level: int | None = None) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where mpp is calculated

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_power(self, wsi, level: int | None = None) -> float:
        """
        Returns the magnification power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where magnification power is calculated

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
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
        self, wsi, patch: NdarrayOrTensor, location: tuple[int, int], size: tuple[int, int], level: int
    ) -> dict:
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
        metadata: dict = {
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
        location: tuple[int, int] = (0, 0),
        size: tuple[int, int] | None = None,
        level: int | None = None,
        mpp: float | tuple[float, float] | None = None,
        power: int | None = None,
        mode: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Verifies inputs, extracts patches from WSI image and generates metadata, and return them.

        Args:
            wsi: a whole slide image object loaded from a file or a list of such objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame. Defaults to (0, 0).
            size: (height, width) tuple giving the patch size at the given level (`level`).
                If not provided or None, it is set to the full image size at the given level.
            level: the level number. Defaults to 0
            mpp: micron per pixel
            power: objective power
            dtype: the data type of output image
            mode: the output image mode, 'RGB' or 'RGBA'


        Returns:
            a tuples, where the first element is an image patch [CxHxW] or stack of patches,
                and second element is a dictionary of metadata
        """
        if mode is None:
            mode = self.mode
        patch_list: list = []
        metadata_list: list = []

        # CuImage object is iterable, so ensure_tuple won't work on single object
        if not isinstance(wsi, (list, tuple)):
            wsi = (wsi,)
        for each_wsi in ensure_tuple(wsi):
            # get the valid level based on resolution info
            level = self._get_valid_level(each_wsi, level, mpp, power)

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

            # Get numpy dtype if it is not already.
            dtype_np = dtype_torch_to_numpy(self.dtype) if isinstance(self.dtype, torch.dtype) else self.dtype
            # Extract a patch or the entire image
            patch: NdarrayOrTensor
            patch = self._get_patch(each_wsi, location=location, size=size, level=level, dtype=dtype_np, mode=mode)

            # Convert the patch to torch.Tensor if dtype is torch
            if isinstance(self.dtype, torch.dtype) or (
                self.device is not None and torch.device(self.device).type == "cuda"
            ):
                # Ensure dtype is torch.dtype if the device is not "cpu"
                dtype_torch = (
                    dtype_numpy_to_torch(self.dtype) if not isinstance(self.dtype, torch.dtype) else self.dtype
                )
                # Copy the numpy array if it is not writable
                if patch.flags["WRITEABLE"]:
                    patch = torch.as_tensor(patch, dtype=dtype_torch, device=self.device)
                else:
                    patch = torch.tensor(patch, dtype=dtype_torch, device=self.device)

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

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
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
        dtype: the data type of output image. Defaults to `np.uint8`.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        num_workers: number of workers for multi-thread image loading (cucim backend only).
        kwargs: additional arguments to be passed to the backend library

    """

    supported_backends = ["cucim", "openslide", "tifffile"]

    def __init__(
        self,
        backend="cucim",
        level: int | None = None,
        mpp: float | tuple[float, float] | None = None,
        power: int | None = None,
        channel_dim: int = 0,
        dtype: DtypeLike | torch.dtype = np.uint8,
        device: torch.device | str | None = None,
        mode: str = "RGB",
        mpp_rtol: float = 0.05,
        mpp_atol: float = 0.0,
        power_rtol: float = 0.05,
        power_atol: float = 0.0,
        **kwargs,
    ):
        self.backend = backend.lower()
        self.reader: CuCIMWSIReader | OpenSlideWSIReader | TiffFileWSIReader
        if self.backend == "cucim":
            self.reader = CuCIMWSIReader(
                level=level,
                mpp=mpp,
                power=power,
                channel_dim=channel_dim,
                dtype=dtype,
                device=device,
                mode=mode,
                mpp_rtol=mpp_rtol,
                mpp_atol=mpp_atol,
                power_rtol=power_rtol,
                power_atol=power_atol,
                **kwargs,
            )
        elif self.backend == "openslide":
            self.reader = OpenSlideWSIReader(
                level=level,
                mpp=mpp,
                power=power,
                channel_dim=channel_dim,
                dtype=dtype,
                device=device,
                mode=mode,
                mpp_rtol=mpp_rtol,
                mpp_atol=mpp_atol,
                power_rtol=power_rtol,
                power_atol=power_atol,
                **kwargs,
            )
        elif self.backend == "tifffile":
            self.reader = TiffFileWSIReader(
                level=level,
                mpp=mpp,
                power=power,
                channel_dim=channel_dim,
                dtype=dtype,
                device=device,
                mode=mode,
                mpp_rtol=mpp_rtol,
                mpp_atol=mpp_atol,
                power_rtol=power_rtol,
                power_atol=power_atol,
                **kwargs,
            )
        else:
            raise ValueError(
                f"The supported backends are cucim, openslide, and tifffile but '{self.backend}' was given."
            )
        self.supported_suffixes = self.reader.supported_suffixes
        self.level = self.reader.level
        self.channel_dim = self.reader.channel_dim
        self.dtype = self.reader.dtype
        self.device = self.reader.device
        self.mode = self.reader.mode
        self.kwargs = self.reader.kwargs
        self.metadata = self.reader.metadata
        self.mpp = self.reader.mpp
        self.power = self.reader.power
        self.mpp_rtol = self.reader.mpp_rtol
        self.mpp_atol = self.reader.mpp_atol
        self.power_rtol = self.reader.power_rtol
        self.power_atol = self.reader.power_atol

    def get_level_count(self, wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return self.reader.get_level_count(wsi)

    def get_size(self, wsi, level: int | None = None) -> tuple[int, int]:
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

    def get_downsample_ratio(self, wsi, level: int | None = None) -> float:
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

    def get_mpp(self, wsi, level: int | None = None) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where mpp calculated.
                If not provided the default level (from `self.level`) will be used.

        """
        if level is None:
            level = self.level

        return self.reader.get_mpp(wsi, level)

    def get_power(self, wsi, level: int | None = None) -> int:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where the magnification power is calculated.
                If not provided the default level (from `self.level`) will be used.

        """
        if level is None:
            level = self.level

        return self.reader.get_power(wsi, level)

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
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

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
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
        dtype: the data type of output image. Defaults to `np.uint8`.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        num_workers: number of workers for multi-thread image loading
        kwargs: additional args for `cucim.CuImage` module:
            https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "cucim"

    def __init__(
        self,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_workers = num_workers

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return wsi.resolutions["level_count"]  # type: ignore

    def get_size(self, wsi, level: int | None = None) -> tuple[int, int]:
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

    def get_downsample_ratio(self, wsi, level: int | None = None) -> float:
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

    def get_mpp(self, wsi, level: int | None = None) -> tuple[float, float]:
        """
        Returns the micro-per-pixel resolution of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where mpp is calculated. If not provided the default level (from `self.level`)
                will be used.

        """
        if level is None:
            level = self.level

        factor = float(wsi.resolutions["level_downsamples"][level])
        return (wsi.metadata["cucim"]["spacing"][1] * factor, wsi.metadata["cucim"]["spacing"][0] * factor)

    def get_power(self, wsi, level: int | None = None) -> int:
        """
        Returns the magnification power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where magnification power is calculated.
                If not provided the default level (from `self.level`) will be used.

        """
        if level is None:
            level = self.level

        if "aperio" in wsi.metadata:
            objective_power = wsi.metadata["aperio"].get("AppMag")
            if objective_power:
                downsample_ratio = self.get_downsample_ratio(wsi, level)
                return round(float(objective_power) / downsample_ratio)

        raise ValueError(
            "Objective power can only be obtained for Aperio images using CuCIM."
            "Please use `level` (or `mpp`) instead, or try OpenSlide backend."
        )

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.
                For more details look at https://github.com/rapidsai/cucim/blob/main/cpp/include/cucim/cuimage.h

        Returns:
            whole slide image object or list of such objects

        """
        cuimage_cls, _ = optional_import("cucim", name="CuImage")
        wsi_list: list = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = cuimage_cls(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
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
        dtype: the data type of output image. Defaults to `np.uint8`.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        kwargs: additional args for `openslide.OpenSlide` module.

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "openslide"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return wsi.level_count  # type: ignore

    def get_size(self, wsi, level: int | None = None) -> tuple[int, int]:
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

    def get_downsample_ratio(self, wsi, level: int | None = None) -> float:
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

    def get_mpp(self, wsi, level: int | None = None) -> tuple[float, float]:
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
        elif unit == "millimeter":
            factor = 1000.0
        elif unit == "micrometer":
            factor = 1.0
        elif unit == "inch":
            factor = 25400.0
        else:
            raise ValueError(f"The resolution unit is not a valid tiff resolution: {unit}")

        factor *= wsi.level_downsamples[level]
        return (factor / float(wsi.properties["tiff.YResolution"]), factor / float(wsi.properties["tiff.XResolution"]))

    def get_power(self, wsi, level: int | None = None) -> int:
        """
        Returns the magnification power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where magnification power is calculated.
                If not provided the default level (from `self.level`) will be used.

        """
        if level is None:
            level = self.level

        objective_power = wsi.properties.get("openslide.objective-power")
        if objective_power:
            downsample_ratio = self.get_downsample_ratio(wsi, level)
            return int(round(objective_power / downsample_ratio))

        raise ValueError("Objective power cannot be obtained for this file. Please use `level` (or `mpp`) instead.")

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.

        Returns:
            whole slide image object or list of such objects

        """
        wsi_list: list = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = OpenSlide(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
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
        dtype: the data type of output image. Defaults to `np.uint8`.
        device: target device to put the extracted patch. Note that if device is "cuda"",
            the output will be converted to torch tenor and sent to the gpu even if the dtype is numpy.
        mode: the output image color mode, "RGB" or "RGBA". Defaults to "RGB".
        kwargs: additional args for `tifffile.TiffFile` module.

    """

    supported_suffixes = ["tif", "tiff", "svs"]
    backend = "tifffile"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_level_count(wsi) -> int:
        """
        Returns the number of levels in the whole slide image.

        Args:
            wsi: a whole slide image object loaded from a file

        """
        return len(wsi.pages)

    def get_size(self, wsi, level: int | None = None) -> tuple[int, int]:
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

    def get_downsample_ratio(self, wsi, level: int | None = None) -> float:
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

    def get_mpp(self, wsi, level: int | None = None) -> tuple[float, float]:
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

    def get_power(self, wsi, level: int | None = None) -> int:
        """
        Returns the magnification power of the whole slide image at a given level.

        Args:
            wsi: a whole slide image object loaded from a file
            level: the level number where magnification power is calculated.
                If not provided the default level (from `self.level`) will be used.

        """
        raise ValueError(
            "Objective power cannot be obtained from TiffFile object."
            "Please use `level` (or `mpp`) instead, or try other backends."
        )

    def read(self, data: Sequence[PathLike] | PathLike | np.ndarray, **kwargs):
        """
        Read whole slide image objects from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args that overrides `self.kwargs` for existing keys.

        Returns:
            whole slide image object or list of such objects

        """
        wsi_list: list = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for filename in filenames:
            wsi = TiffFile(filename, **kwargs_)
            wsi_list.append(wsi)

        return wsi_list if len(filenames) > 1 else wsi_list[0]

    def _get_patch(
        self, wsi, location: tuple[int, int], size: tuple[int, int], level: int, dtype: DtypeLike, mode: str
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
        patch = wsi_image[location_[0] : location_[0] + size[0], location_[1] : location_[1] + size[1], :]

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
