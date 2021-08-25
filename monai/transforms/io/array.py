# Copyright 2020 - 2021 MONAI Consortium
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

import inspect
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import DtypeLike
from monai.data.image_reader import ImageReader, ITKReader, NibabelReader, NumpyReader, PILReader
from monai.data.nifti_saver import NiftiSaver
from monai.data.png_saver import PNGSaver
from monai.transforms.transform import Transform
from monai.utils import GridSampleMode, GridSamplePadMode
from monai.utils import ImageMetaKey as Key
from monai.utils import InterpolateMode, ensure_tuple, optional_import
from monai.utils.module import look_up_option

nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")

__all__ = ["LoadImage", "SaveImage", "SUPPORTED_READERS"]

SUPPORTED_READERS = {
    "itkreader": ITKReader,
    "numpyreader": NumpyReader,
    "pilreader": PILReader,
    "nibabelreader": NibabelReader,
}


def switch_endianness(data, new="<"):
    """
    Convert the input `data` endianness to `new`.

    Args:
        data: input to be converted.
        new: the target endianness, currently support "<" or ">".
    """
    if isinstance(data, np.ndarray):
        # default to system endian
        sys_native = "<" if (sys.byteorder == "little") else ">"
        current_ = sys_native if data.dtype.byteorder not in ("<", ">") else data.dtype.byteorder
        if new not in ("<", ">"):
            raise NotImplementedError(f"Not implemented option new={new}.")
        if current_ != new:
            data = data.byteswap().newbyteorder(new)
    elif isinstance(data, tuple):
        data = tuple(switch_endianness(x, new) for x in data)
    elif isinstance(data, list):
        data = [switch_endianness(x, new) for x in data]
    elif isinstance(data, dict):
        data = {k: switch_endianness(v, new) for k, v in data.items()}
    elif not isinstance(data, (bool, str, float, int, type(None))):
        raise RuntimeError(f"Unknown type: {type(data).__name__}")
    return data


class LoadImage(Transform):
    """
    Load image file or files from provided path based on reader.
    If reader is not specified, this class automatically chooses readers
    based on the supported suffixes and in the following order:

        - User-specified reader at runtime when calling this loader.
        - User-specified reader in the constructor of `LoadImage`.
        - Readers from the last to the first in the registered list.
        - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
          (npz, npy -> NumpyReader), (others -> ITKReader).

    See also:

        - tutorial: https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb

    """

    def __init__(self, reader=None, image_only: bool = False, dtype: DtypeLike = np.float32, *args, **kwargs) -> None:
        """
        Args:
            reader: reader to load image file and meta data

                - if `reader` is None, a default set of `SUPPORTED_READERS` will be used.
                - if `reader` is a string, the corresponding item in `SUPPORTED_READERS` will be used,
                  and a reader instance will be constructed with the `*args` and `**kwargs` parameters.
                  the supported reader names are: "nibabelreader", "pilreader", "itkreader", "numpyreader".
                - if `reader` is a reader class/instance, it will be registered to this loader accordingly.

            image_only: if True return only the image volume, otherwise return image data array and header dict.
            dtype: if not None convert the loaded image to this data type.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.

        Note:

            - The transform returns an image data array if `image_only` is True,
              or a tuple of two elements containing the data array, and the meta data in a dictionary format otherwise.
            - If `reader` is specified, the loader will attempt to use the specified readers and the default supported
              readers. This might introduce overheads when handling the exceptions of trying the incompatible loaders.
              In this case, it is therefore recommended to set the most appropriate reader as
              the last item of the `reader` parameter.

        """

        self.auto_select = reader is None
        self.image_only = image_only
        self.dtype = dtype

        self.readers: List[ImageReader] = []
        for r in SUPPORTED_READERS:  # set predefined readers as default
            try:
                self.register(SUPPORTED_READERS[r](*args, **kwargs))
            except TypeError:  # the reader doesn't have the corresponding args/kwargs
                logging.getLogger(self.__class__.__name__).debug(
                    f"{r} is not supported with the given parameters {args} {kwargs}."
                )
                self.register(SUPPORTED_READERS[r]())
        if reader is None:
            return  # no user-specified reader, no need to register

        for _r in ensure_tuple(reader):
            if isinstance(_r, str):
                the_reader = look_up_option(_r.lower(), SUPPORTED_READERS)
                try:
                    self.register(the_reader(*args, **kwargs))
                except TypeError:  # the reader doesn't have the corresponding args/kwargs
                    warnings.warn(f"{r} is not supported with the given parameters {args} {kwargs}.")
                    self.register(the_reader())
            elif inspect.isclass(_r):
                self.register(_r(*args, **kwargs))
            else:
                self.register(_r)  # reader instance, ignoring the constructor args/kwargs
        return

    def register(self, reader: ImageReader):
        """
        Register image reader to load image file and meta data.

        Args:
            reader: reader instance to be registered with this loader.

        """
        if not isinstance(reader, ImageReader):
            warnings.warn(f"Preferably the reader should inherit ImageReader, but got {type(reader)}.")
        self.readers.append(reader)

    def __call__(self, filename: Union[Sequence[str], str, Path, Sequence[Path]], reader: Optional[ImageReader] = None):
        """
        Load image file and meta data from the given filename(s).
        If `reader` is not specified, this class automatically chooses readers based on the
        reversed order of registered readers `self.readers`.

        Args:
            filename: path file or file-like object or a list of files.
                will save the filename to meta_data with key `filename_or_obj`.
                if provided a list of files, use the filename of first file.
            reader: runtime reader to load image file and meta data.

        """
        filename = tuple(str(s) for s in ensure_tuple(filename))  # allow Path objects
        img = None
        if reader is not None:
            img = reader.read(filename)  # runtime specified reader
        else:
            for reader in self.readers[::-1]:
                if self.auto_select:  # rely on the filename extension to choose the reader
                    if reader.verify_suffix(filename):
                        img = reader.read(filename)
                        break
                else:  # try the user designated readers
                    try:
                        img = reader.read(filename)
                    except Exception as e:
                        logging.getLogger(self.__class__.__name__).debug(
                            f"{reader.__class__.__name__}: unable to load {filename}.\n" f"Error: {e}"
                        )
                    else:
                        break

        if img is None or reader is None:
            raise RuntimeError(
                f"can not find a suitable reader for file: {filename}.\n"
                "    Please install the reader libraries, see also the installation instructions:\n"
                "    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies.\n"
                f"   The current registered: {self.readers}.\n"
            )

        img_array, meta_data = reader.get_data(img)
        img_array = img_array.astype(self.dtype)

        if self.image_only:
            return img_array
        meta_data[Key.FILENAME_OR_OBJ] = ensure_tuple(filename)[0]
        # make sure all elements in metadata are little endian
        meta_data = switch_endianness(meta_data, "<")

        return img_array, meta_data


class SaveImage(Transform):
    """
    Save transformed data into files, support NIfTI and PNG formats.
    It can work for both numpy array and PyTorch Tensor in both preprocessing transform
    chain and postprocessing transform chain.
    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the input image name is extracted from the provided meta data dictionary.
    If no meta data provided, use index from 0 as the filename prefix.
    It can also save a list of PyTorch Tensor or numpy array without `batch dim`.

    Note: image should be channel-first shape: [C,H,W,[D]].

    Args:
        output_dir: output image directory.
        output_postfix: a string appended to all output file names, default to `trans`.
        output_ext: output file extension name, available extensions: `.nii.gz`, `.nii`, `.png`.
        resample: whether to resample before saving the data array.
            if saving PNG format image, based on the `spatial_shape` from metadata.
            if saving NIfTI format image, based on the `original_affine` from metadata.
        mode: This option is used when ``resample = True``. Defaults to ``"nearest"``.

            - NIfTI files {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.

            - NIfTI files {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files
                This option is ignored.

        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
            it's used for PNG format only.
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
            it's used for NIfTI format only.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
            it's used for NIfTI format only.
        squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D also ==1, it will be saved as (H,W). If false,
            image will always be saved as (H,W,D,C).
            it's used for NIfTI format only.
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. it's used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names. for example:
            input_file_name: /foo/bar/test1/image.nii,
            output_postfix: seg
            output_ext: nii.gz
            output_dir: /output,
            data_root_dir: /foo/bar,
            output will be: /output/test1/image/image_seg.nii.gz
        separate_folder: whether to save every file in a separate folder, for example: if input filename is
            `image.nii`, postfix is `seg` and folder_path is `output`, if `True`, save as:
            `output/image/image_seg.nii`, if `False`, save as `output/image_seg.nii`. default to `True`.
        print_log: whether to print log about the saved file path, etc. default to `True`.

    """

    def __init__(
        self,
        output_dir: Union[Path, str] = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
    ) -> None:
        self.saver: Union[NiftiSaver, PNGSaver]
        if output_ext in {".nii.gz", ".nii"}:
            self.saver = NiftiSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=GridSampleMode(mode),
                padding_mode=padding_mode,
                dtype=dtype,
                output_dtype=output_dtype,
                squeeze_end_dims=squeeze_end_dims,
                data_root_dir=data_root_dir,
                separate_folder=separate_folder,
                print_log=print_log,
            )
        elif output_ext == ".png":
            self.saver = PNGSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=InterpolateMode(mode),
                scale=scale,
                data_root_dir=data_root_dir,
                separate_folder=separate_folder,
                print_log=print_log,
            )
        else:
            raise ValueError(f"unsupported output extension: {output_ext}.")

    def __call__(self, img: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None):
        """
        Args:
            img: target data content that save into file.
            meta_data: key-value pairs of meta_data corresponding to the data.

        """
        self.saver.save(img, meta_data)

        return img
