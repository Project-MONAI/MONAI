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

from typing import List, Dict, Optional, Sequence, Union

import numpy as np
import torch

from monai.data.nifti_saver import NiftiSaver
from monai.data.png_saver import PNGSaver
from monai.data.image_reader import ImageReader, ITKReader, NibabelReader, NumpyReader, PILReader
from monai.transforms.compose import Transform
from monai.utils import (
    ensure_tuple,
    optional_import,
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
)

nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")

__all__ = ["LoadImage", "SaveImage"]


class LoadImage(Transform):
    """
    Load image file or files from provided path based on reader.
    Automatically choose readers based on the supported suffixes and in below order:
    - User specified reader at runtime when call this loader.
    - Registered readers from the latest to the first in list.
    - Default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
    (npz, npy -> NumpyReader), (others -> ITKReader).

    """

    def __init__(
        self,
        reader: Optional[Union[ImageReader, str]] = None,
        image_only: bool = False,
        dtype: np.dtype = np.float32,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            reader: register reader to load image file and meta data, if None, still can register readers
                at runtime or use the default readers. If a string of reader name provided, will construct
                a reader object with the `*args` and `**kwargs` parameters, supported reader name: "NibabelReader",
                "PILReader", "ITKReader", "NumpyReader"
            image_only: if True return only the image volume, otherwise return image data array and header dict.
            dtype: if not None convert the loaded image to this data type.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.

        Note:
            The transform returns image data array if `image_only` is True,
            or a tuple of two elements containing the data array, and the meta data in a dict format otherwise.

        """
        # set predefined readers as default
        self.readers: List[ImageReader] = [ITKReader(), NumpyReader(), PILReader(), NibabelReader()]
        if reader is not None:
            if isinstance(reader, str):
                supported_readers = {
                    "nibabelreader": NibabelReader,
                    "pilreader": PILReader,
                    "itkreader": ITKReader,
                    "numpyreader": NumpyReader,
                }
                reader = reader.lower()
                if reader not in supported_readers:
                    raise ValueError(f"unsupported reader type: {reader}, available options: {supported_readers}.")
                self.register(supported_readers[reader](*args, **kwargs))
            else:
                self.register(reader)

        self.image_only = image_only
        self.dtype = dtype

    def register(self, reader: ImageReader) -> List[ImageReader]:
        """
        Register image reader to load image file and meta data, latest registered reader has higher priority.
        Return all the registered image readers.

        Args:
            reader: registered reader to load image file and meta data based on suffix,
                if all registered readers can't match suffix at runtime, use the default readers.

        """
        if not isinstance(reader, ImageReader):
            raise ValueError(f"reader must be ImageReader object, but got {type(reader)}.")
        self.readers.append(reader)
        return self.readers

    def __call__(
        self,
        filename: Union[Sequence[str], str],
        reader: Optional[ImageReader] = None,
    ):
        """
        Args:
            filename: path file or file-like object or a list of files.
                will save the filename to meta_data with key `filename_or_obj`.
                if provided a list of files, use the filename of first file.
            reader: runtime reader to load image file and meta data.

        """
        if reader is None or not reader.verify_suffix(filename):
            for r in reversed(self.readers):
                if r.verify_suffix(filename):
                    reader = r
                    break

        if reader is None:
            raise RuntimeError(f"can not find suitable reader for this file: {filename}.")

        img = reader.read(filename)
        img_array, meta_data = reader.get_data(img)
        img_array = img_array.astype(self.dtype)

        if self.image_only:
            return img_array
        meta_data["filename_or_obj"] = ensure_tuple(filename)[0]
        return img_array, meta_data


class SaveImage(Transform):
    """
    Save transformed data into files, support NIfTI and PNG formats.
    It can work for both numpy array and PyTorch Tensor in both pre-transform chain
    and post transform chain.

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
        save_batch: whether the import image is a batch data, default to `False`.
            usually pre-transforms run for channel first data, while post-transforms run for batch data.

    """
    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: Optional[np.dtype] = np.float64,
        output_dtype: Optional[np.dtype] = np.float32,
        save_batch: bool = False,
    ) -> None:
        self.saver: Union[NiftiSaver, PNGSaver]
        if output_ext in (".nii.gz", ".nii"):
            self.saver = NiftiSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=GridSampleMode(mode),
                padding_mode=padding_mode,
                dtype=dtype,
                output_dtype=output_dtype,
            )
        elif output_ext == ".png":
            self.saver = PNGSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=InterpolateMode(mode),
                scale=scale,
            )
        self.save_batch = save_batch

    def __call__(self, img: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None):
        if self.save_batch:
            self.saver.save_batch(img, meta_data)
        else:
            self.saver.save(img, meta_data)
