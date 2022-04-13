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

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai.config import DtypeLike, KeysCollection, PathLike
from monai.data.utils import correct_nifti_header_if_necessary, is_supported_format, orientation_ras_lps
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import ensure_tuple, ensure_tuple_rep, optional_import, require_pkg

if TYPE_CHECKING:
    import itk
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Image
    from PIL import Image as PILImage

    has_itk = has_nib = has_pil = True
else:
    itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
    nib, has_nib = optional_import("nibabel")
    Nifti1Image, _ = optional_import("nibabel.nifti1", name="Nifti1Image")
    PILImage, has_pil = optional_import("PIL.Image")

OpenSlide, _ = optional_import("openslide", name="OpenSlide")
CuImage, _ = optional_import("cucim", name="CuImage")
TiffFile, _ = optional_import("tifffile", name="TiffFile")

__all__ = ["ImageReader", "ITKReader", "NibabelReader", "NumpyReader", "PILReader"]


class ImageReader(ABC):
    """
    An abstract class defines APIs to load image files.

    Typical usage of an implementation of this class is:

    .. code-block:: python

        image_reader = MyImageReader()
        img_obj = image_reader.read(path_to_image)
        img_data, meta_data = image_reader.get_data(img_obj)

    - The `read` call converts image filenames into image objects,
    - The `get_data` call fetches the image data, as well as meta data.
    - A reader should implement `verify_suffix` with the logic of checking the input filename
      by the filename extensions.

    """

    @abstractmethod
    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified `filename` is supported by the current reader.
        This method should return True if the reader is able to read the format suggested by the
        `filename`.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs) -> Union[Sequence[Any], Any]:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        """
        Extract data array and meta data from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of meta data.

        Args:
            img: an image object loaded from an image file or a list of image objects.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


def _copy_compatible_dict(from_dict: Dict, to_dict: Dict):
    if not isinstance(to_dict, dict):
        raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
    if not to_dict:
        for key in from_dict:
            datum = from_dict[key]
            if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
                continue
            to_dict[key] = datum
    else:
        affine_key, shape_key = "affine", "spatial_shape"
        if affine_key in from_dict and not np.allclose(from_dict[affine_key], to_dict[affine_key]):
            raise RuntimeError(
                "affine matrix of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
            )
        if shape_key in from_dict and not np.allclose(from_dict[shape_key], to_dict[shape_key]):
            raise RuntimeError(
                "spatial_shape of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )


def _stack_images(image_list: List, meta_dict: Dict):
    if len(image_list) <= 1:
        return image_list[0]
    if meta_dict.get("original_channel_dim", None) not in ("no_channel", None):
        channel_dim = int(meta_dict["original_channel_dim"])
        return np.concatenate(image_list, axis=channel_dim)
    # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
    meta_dict["original_channel_dim"] = 0
    return np.stack(image_list, axis=0)


@require_pkg(pkg_name="itk")
class ITKReader(ImageReader):
    """
    Load medical images based on ITK library.
    All the supported image formats can be found at:
    https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/IO
    The loaded data array will be in C order, for example, a 3D image NumPy
    array index order will be `CDWH`.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the meta data, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `-1`.

                - Nifti file is usually "channel last", so there is no need to specify this argument.
                - PNG file usually has `GetNumberOfComponentsPerPixel()==3`, so there is no need to specify this argument.

        series_name: the name of the DICOM series if there are multiple ones.
            used when loading DICOM series.
        reverse_indexing: whether to use a reversed spatial indexing convention for the returned data array.
            If ``False``, the spatial indexing follows the numpy convention;
            otherwise, the spatial indexing convention is reversed to be compatible with ITK. Default is ``False``.
            This option does not affect the metadata.
        series_meta: whether to load the metadata of the DICOM series (using the metadata from the first slice).
            This flag is checked only when loading DICOM series. Default is ``False``.
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to ``True``.
            Set to ``True`` to be consistent with ``NibabelReader``, otherwise the affine matrix remains in the ITK convention.
        kwargs: additional args for `itk.imread` API. more details about available args:
            https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

    """

    def __init__(
        self,
        channel_dim: Optional[int] = None,
        series_name: str = "",
        reverse_indexing: bool = False,
        series_meta: bool = False,
        affine_lps_to_ras: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.channel_dim = channel_dim
        self.series_name = series_name
        self.reverse_indexing = reverse_indexing
        self.series_meta = series_meta
        self.affine_lps_to_ras = affine_lps_to_ras

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by ITK reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        return has_itk

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        If passing directory path instead of file path, will treat it as DICOM images series and read.
        Note that the returned object is ITK image object or list of ITK image objects.

        Args:
            data: file name or a list of file names to read,
            kwargs: additional args for `itk.imread` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itkExtras.py

        """
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            name = f"{name}"
            if Path(name).is_dir():
                # read DICOM series
                # https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage
                names_generator = itk.GDCMSeriesFileNames.New()
                names_generator.SetUseSeriesDetails(True)
                names_generator.AddSeriesRestriction("0008|0021")  # Series Date
                names_generator.SetDirectory(name)
                series_uid = names_generator.GetSeriesUIDs()

                if len(series_uid) < 1:
                    raise FileNotFoundError(f"no DICOMs in: {name}.")
                if len(series_uid) > 1:
                    warnings.warn(f"the directory: {name} contains more than one DICOM series.")
                series_identifier = series_uid[0] if not self.series_name else self.series_name
                name = names_generator.GetFileNames(series_identifier)

                _obj = itk.imread(name, **kwargs_)
                if self.series_meta:
                    _reader = itk.ImageSeriesReader.New(FileNames=name)
                    _reader.Update()
                    _meta = _reader.GetMetaDataDictionaryArray()
                    if len(_meta) > 0:
                        # TODO: using the first slice's meta. this could be improved to filter unnecessary tags.
                        _obj.SetMetaDataDictionary(_meta[0])
                img_.append(_obj)
            else:
                img_.append(itk.imread(name, **kwargs_))
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the meta data of the first image is used to represent the output meta data.

        Args:
            img: an ITK image object loaded from an image file or a list of ITK image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            data = self._get_array_data(i)
            img_array.append(data)
            header = self._get_meta_dict(i)
            header["original_affine"] = self._get_affine(i, self.affine_lps_to_ras)
            header["affine"] = header["original_affine"].copy()
            header["spatial_shape"] = self._get_spatial_shape(i)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            else:
                header["original_channel_dim"] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get all the meta data of the image and convert to dict type.

        Args:
            img: an ITK image object loaded from an image file.

        """
        img_meta_dict = img.GetMetaDataDictionary()
        meta_dict = {key: img_meta_dict[key] for key in img_meta_dict.GetKeys() if not key.startswith("ITK_")}

        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        return meta_dict

    def _get_affine(self, img, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: an ITK image object loaded from an image file.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """
        direction = itk.array_from_matrix(img.GetDirection())
        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())

        direction = np.asarray(direction)
        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of `img`.

        Args:
            img: an ITK image object loaded from an image file.

        """
        sr = itk.array_from_matrix(img.GetDirection()).shape[0]
        sr = max(min(sr, 3), 1)
        _size = list(itk.size(img))
        if self.channel_dim is not None:
            _size.pop(self.channel_dim)
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Following PyTorch conventions, the returned array data has contiguous channels,
        e.g. for an RGB image, all red channel image pixels are contiguous in memory.
        The last axis of the returned array is the channel axis.

        See also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Modules/Bridge/NumPy/wrapping/PyBuffer.i.in

        Args:
            img: an ITK image object loaded from an image file.

        """
        np_img = itk.array_view_from_image(img, keep_axes=False)
        if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
            return np_img if self.reverse_indexing else np_img.T
        # handling multi-channel images
        return np_img if self.reverse_indexing else np.moveaxis(np_img.T, 0, -1)


@require_pkg(pkg_name="nibabel")
class NibabelReader(ImageReader):
    """
    Load NIfTI format images based on Nibabel library.

    Args:
        as_closest_canonical: if True, load the image as closest to canonical axis format.
        squeeze_non_spatial_dims: if True, non-spatial singletons will be squeezed, e.g. (256,256,1,3) -> (256,256,3)
        channel_dim: the channel dimension of the input image, default is None.
            this is used to set original_channel_dim in the meta data, EnsureChannelFirstD reads this field.
            if None, `original_channel_dim` will be either `no_channel` or `-1`.
            most Nifti files are usually "channel last", no need to specify this argument for them.
        dtype: dtype of the output data array when loading with Nibabel library.
        kwargs: additional args for `nibabel.load` API. more details about available args:
            https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

    """

    def __init__(
        self,
        channel_dim: Optional[int] = None,
        as_closest_canonical: bool = False,
        squeeze_non_spatial_dims: bool = False,
        dtype: DtypeLike = np.float32,
        **kwargs,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.as_closest_canonical = as_closest_canonical
        self.squeeze_non_spatial_dims = squeeze_non_spatial_dims
        self.dtype = dtype
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by Nibabel reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        suffixes: Sequence[str] = ["nii", "nii.gz"]
        return has_nib and is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is Nibabel image object or list of Nibabel image objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `nibabel.load` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = nib.load(name, **kwargs_)
            img = correct_nifti_header_if_necessary(img)
            img_.append(img)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the meta data of the first image is used to present the output meta data.

        Args:
            img: a Nibabel image object loaded from an image file or a list of Nibabel image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            header = self._get_meta_dict(i)
            header["affine"] = self._get_affine(i)
            header["original_affine"] = self._get_affine(i)
            header["as_closest_canonical"] = self.as_closest_canonical
            if self.as_closest_canonical:
                i = nib.as_closest_canonical(i)
                header["affine"] = self._get_affine(i)
            header["spatial_shape"] = self._get_spatial_shape(i)
            data = self._get_array_data(i)
            if self.squeeze_non_spatial_dims:
                for d in range(len(data.shape), len(header["spatial_shape"]), -1):
                    if data.shape[d - 1] == 1:
                        data = data.squeeze(axis=d - 1)
            img_array.append(data)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            else:
                header["original_channel_dim"] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        # swap to little endian as PyTorch doesn't support big endian
        try:
            header = img.header.as_byteswapped("<")
        except ValueError:
            header = img.header
        return dict(header)

    def _get_affine(self, img):
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        return np.array(img.affine, copy=True)

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        # swap to little endian as PyTorch doesn't support big endian
        try:
            header = img.header.as_byteswapped("<")
        except ValueError:
            header = img.header
        dim = header.get("dim", None)
        if dim is None:
            dim = header.get("dims")  # mgh format?
            dim = np.insert(dim, 0, 3)
        ndim = dim[0]
        size = list(dim[1:])
        if self.channel_dim is not None:
            size.pop(self.channel_dim)
        spatial_rank = max(min(ndim, 3), 1)
        return np.asarray(size[:spatial_rank])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        _array = np.array(img.get_fdata(dtype=self.dtype))
        img.uncache()
        return _array


class NumpyReader(ImageReader):
    """
    Load NPY or NPZ format data based on Numpy library, they can be arrays or pickled objects.
    A typical usage is to load the `mask` data for classification task.
    It can load part of the npz file with specified `npz_keys`.

    Args:
        npz_keys: if loading npz file, only load the specified keys, if None, load all the items.
            stack the loaded items together to construct a new first dimension.
        channel_dim: if not None, explicitly specify the channel dim, otherwise, treat the array as no channel.
        kwargs: additional args for `numpy.load` API except `allow_pickle`. more details about available args:
            https://numpy.org/doc/stable/reference/generated/numpy.load.html

    """

    def __init__(self, npz_keys: Optional[KeysCollection] = None, channel_dim: Optional[int] = None, **kwargs):
        super().__init__()
        if npz_keys is not None:
            npz_keys = ensure_tuple(npz_keys)
        self.npz_keys = npz_keys
        self.channel_dim = channel_dim
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by Numpy reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["npz", "npy"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of data files
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is Numpy array or list of Numpy arrays.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `numpy.load` API except `allow_pickle`, will override `self.kwargs` for existing keys.
                More details about available args:
                https://numpy.org/doc/stable/reference/generated/numpy.load.html

        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = np.load(name, allow_pickle=True, **kwargs_)
            if Path(name).name.endswith(".npz"):
                # load expected items from NPZ file
                npz_keys = [f"arr_{i}" for i in range(len(img))] if self.npz_keys is None else self.npz_keys
                for k in npz_keys:
                    img_.append(img[k])
            else:
                img_.append(img)

        return img_ if len(img_) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the meta data of the first image is used to represent the output meta data.

        Args:
            img: a Numpy array loaded from a file or a list of Numpy arrays.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}
        if isinstance(img, np.ndarray):
            img = (img,)

        for i in ensure_tuple(img):
            header = {}
            if isinstance(i, np.ndarray):
                # if `channel_dim` is None, can not detect the channel dim, use all the dims as spatial_shape
                spatial_shape = np.asarray(i.shape)
                if isinstance(self.channel_dim, int):
                    spatial_shape = np.delete(spatial_shape, self.channel_dim)
                header["spatial_shape"] = spatial_shape
            img_array.append(i)
            header["original_channel_dim"] = self.channel_dim if isinstance(self.channel_dim, int) else "no_channel"
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta


@require_pkg(pkg_name="PIL")
class PILReader(ImageReader):
    """
    Load common 2D image format (supports PNG, JPG, BMP) file or files from provided path.

    Args:
        converter: additional function to convert the image data after `read()`.
            for example, use `converter=lambda image: image.convert("LA")` to convert image format.
        kwargs: additional args for `Image.open` API in `read()`, mode details about available args:
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
    """

    def __init__(self, converter: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.converter = converter
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by PIL reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["png", "jpg", "jpeg", "bmp"]
        return has_pil and is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is PIL image or list of PIL image.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `Image.open` API in `read()`, will override `self.kwargs` for existing keys.
                Mode details about available args:
                https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open

        """
        img_: List[PILImage.Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = PILImage.open(name, **kwargs_)
            if callable(self.converter):
                img = self.converter(img)
            img_.append(img)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It computes `spatial_shape` and stores it in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the meta data of the first image is used to represent the output meta data.
        Note that it will swap axis 0 and 1 after loading the array because the `HW` definition in PIL
        is different from other common medical packages.

        Args:
            img: a PIL Image object loaded from a file or a list of PIL Image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            header = self._get_meta_dict(i)
            header["spatial_shape"] = self._get_spatial_shape(i)
            data = np.moveaxis(np.asarray(i), 0, 1)
            img_array.append(data)
            header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.
        Args:
            img: a PIL Image object loaded from an image file.

        """
        return {"format": img.format, "mode": img.mode, "width": img.width, "height": img.height}

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        return np.asarray((img.width, img.height))
