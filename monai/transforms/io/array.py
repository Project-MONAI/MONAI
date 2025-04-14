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
"""
A collection of "vanilla" transforms for IO functions.
"""
from __future__ import annotations

import inspect
import json
import logging
import sys
import traceback
import warnings
from collections.abc import Sequence
from pathlib import Path
from pydoc import locate
from typing import Callable

import numpy as np
import torch

from monai.config import DtypeLike, NdarrayOrTensor, PathLike
from monai.data import image_writer
from monai.data.folder_layout import FolderLayout, FolderLayoutBase, default_name_formatter
from monai.data.image_reader import (
    ImageReader,
    ITKReader,
    NibabelReader,
    NrrdReader,
    NumpyReader,
    PILReader,
    PydicomReader,
)
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import is_no_channel
from monai.transforms.transform import Transform
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import GridSamplePadMode
from monai.utils import ImageMetaKey as Key
from monai.utils import (
    MetaKeys,
    OptionalImportError,
    convert_to_dst_type,
    ensure_tuple,
    look_up_option,
    optional_import,
)

nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")
nrrd, _ = optional_import("nrrd")
FileLock, has_filelock = optional_import("filelock", name="FileLock")

__all__ = ["LoadImage", "SaveImage", "SUPPORTED_READERS"]

SUPPORTED_READERS = {
    "pydicomreader": PydicomReader,
    "itkreader": ITKReader,
    "nrrdreader": NrrdReader,
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
    if isinstance(data, torch.Tensor):
        device = data.device
        requires_grad: bool = data.requires_grad
        data = (
            torch.from_numpy(switch_endianness(data.cpu().detach().numpy(), new))
            .to(device)
            .requires_grad_(requires_grad=requires_grad)  # type: ignore
        )
    elif isinstance(data, np.ndarray):
        # default to system endian
        sys_native = "<" if (sys.byteorder == "little") else ">"
        current_ = sys_native if data.dtype.byteorder not in ("<", ">") else data.dtype.byteorder
        if new not in ("<", ">"):
            raise NotImplementedError(f"Not implemented option new={new}.")
        if current_ != new:
            data = data.byteswap().view(data.dtype.newbyteorder(new))
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
          (npz, npy -> NumpyReader), (nrrd -> NrrdReader), (DICOM file -> ITKReader).

    Please note that for png, jpg, bmp, and other 2D formats, readers by default swap axis 0 and 1 after
    loading the array with ``reverse_indexing`` set to ``True`` because the spatial axes definition
    for non-medical specific file formats is different from other common medical packages.

    See also:

        - tutorial: https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb

    """

    def __init__(
        self,
        reader=None,
        image_only: bool = True,
        dtype: DtypeLike | None = np.float32,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
        expanduser: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            reader: reader to load image file and metadata
                - if `reader` is None, a default set of `SUPPORTED_READERS` will be used.
                - if `reader` is a string, it's treated as a class name or dotted path
                (such as ``"monai.data.ITKReader"``), the supported built-in reader classes are
                ``"ITKReader"``, ``"NibabelReader"``, ``"NumpyReader"``, ``"PydicomReader"``.
                a reader instance will be constructed with the `*args` and `**kwargs` parameters.
                - if `reader` is a reader class/instance, it will be registered to this loader accordingly.
            image_only: if True return only the image MetaTensor, otherwise return image and header dict.
            dtype: if not None convert the loaded image to this data type.
            ensure_channel_first: if `True` and loaded both image array and metadata, automatically convert
                the image array shape to `channel first`. default to `False`.
            simple_keys: whether to remove redundant metadata keys, default to False for backward compatibility.
            prune_meta_pattern: combined with `prune_meta_sep`, a regular expression used to match and prune keys
                in the metadata (nested dictionary), default to None, no key deletion.
            prune_meta_sep: combined with `prune_meta_pattern`, used to match and prune keys
                in the metadata (nested dictionary). default is ".", see also :py:class:`monai.transforms.DeleteItemsd`.
                e.g. ``prune_meta_pattern=".*_code$", prune_meta_sep=" "`` removes meta keys that ends with ``"_code"``.
            expanduser: if True cast filename to Path and call .expanduser on it, otherwise keep filename as is.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.

        Note:

            - The transform returns a MetaTensor, unless `set_track_meta(False)` has been used, in which case, a
              `torch.Tensor` will be returned.
            - If `reader` is specified, the loader will attempt to use the specified readers and the default supported
              readers. This might introduce overheads when handling the exceptions of trying the incompatible loaders.
              In this case, it is therefore recommended setting the most appropriate reader as
              the last item of the `reader` parameter.

        """

        self.auto_select = reader is None
        self.image_only = image_only
        self.dtype = dtype
        self.ensure_channel_first = ensure_channel_first
        self.simple_keys = simple_keys
        self.pattern = prune_meta_pattern
        self.sep = prune_meta_sep
        self.expanduser = expanduser

        self.readers: list[ImageReader] = []
        for r in SUPPORTED_READERS:  # set predefined readers as default
            try:
                self.register(SUPPORTED_READERS[r](*args, **kwargs))
            except OptionalImportError:
                logging.getLogger(self.__class__.__name__).debug(
                    f"required package for reader {r} is not installed, or the version doesn't match requirement."
                )
            except TypeError:  # the reader doesn't have the corresponding args/kwargs
                logging.getLogger(self.__class__.__name__).debug(
                    f"{r} is not supported with the given parameters {args} {kwargs}."
                )
                self.register(SUPPORTED_READERS[r]())
        if reader is None:
            return  # no user-specified reader, no need to register

        for _r in ensure_tuple(reader):
            if isinstance(_r, str):
                the_reader, has_built_in = optional_import("monai.data", name=f"{_r}")  # search built-in
                if not has_built_in:
                    the_reader = locate(f"{_r}")  # search dotted path
                if the_reader is None:
                    the_reader = look_up_option(_r.lower(), SUPPORTED_READERS)
                try:
                    self.register(the_reader(*args, **kwargs))
                except OptionalImportError:
                    warnings.warn(
                        f"required package for reader {_r} is not installed, or the version doesn't match requirement."
                    )
                except TypeError:  # the reader doesn't have the corresponding args/kwargs
                    warnings.warn(f"{_r} is not supported with the given parameters {args} {kwargs}.")
                    self.register(the_reader())
            elif inspect.isclass(_r):
                self.register(_r(*args, **kwargs))
            else:
                self.register(_r)  # reader instance, ignoring the constructor args/kwargs
        return

    def register(self, reader: ImageReader):
        """
        Register image reader to load image file and metadata.

        Args:
            reader: reader instance to be registered with this loader.

        """
        if not isinstance(reader, ImageReader):
            warnings.warn(f"Preferably the reader should inherit ImageReader, but got {type(reader)}.")
        self.readers.append(reader)

    def __call__(self, filename: Sequence[PathLike] | PathLike, reader: ImageReader | None = None):
        """
        Load image file and metadata from the given filename(s).
        If `reader` is not specified, this class automatically chooses readers based on the
        reversed order of registered readers `self.readers`.

        Args:
            filename: path file or file-like object or a list of files.
                will save the filename to meta_data with key `filename_or_obj`.
                if provided a list of files, use the filename of first file to save,
                and will stack them together as multi-channels data.
                if provided directory path instead of file path, will treat it as
                DICOM images series and read.
            reader: runtime reader to load image file and metadata.

        """
        filename = tuple(
            f"{Path(s).expanduser()}" if self.expanduser else s for s in ensure_tuple(filename)  # allow Path objects
        )
        img, err = None, []
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
                        err.append(traceback.format_exc())
                        logging.getLogger(self.__class__.__name__).debug(e, exc_info=True)
                        logging.getLogger(self.__class__.__name__).info(
                            f"{reader.__class__.__name__}: unable to load {filename}.\n"
                        )
                    else:
                        err = []
                        break

        if img is None or reader is None:
            if isinstance(filename, Sequence) and len(filename) == 1:
                filename = filename[0]
            msg = "\n".join([f"{e}" for e in err])
            raise RuntimeError(
                f"{self.__class__.__name__} cannot find a suitable reader for file: {filename}.\n"
                "    Please install the reader libraries, see also the installation instructions:\n"
                "    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies.\n"
                f"   The current registered: {self.readers}.\n{msg}"
            )
        img_array: NdarrayOrTensor
        img_array, meta_data = reader.get_data(img)
        img_array = convert_to_dst_type(img_array, dst=img_array, dtype=self.dtype)[0]
        if not isinstance(meta_data, dict):
            raise ValueError(f"`meta_data` must be a dict, got type {type(meta_data)}.")
        # make sure all elements in metadata are little endian
        meta_data = switch_endianness(meta_data, "<")

        meta_data[Key.FILENAME_OR_OBJ] = f"{ensure_tuple(filename)[0]}"  # Path obj should be strings for data loader
        img = MetaTensor.ensure_torch_and_prune_meta(
            img_array, meta_data, self.simple_keys, pattern=self.pattern, sep=self.sep
        )
        if self.ensure_channel_first:
            img = EnsureChannelFirst()(img)
        if self.image_only:
            return img
        return img, img.meta if isinstance(img, MetaTensor) else meta_data


class SaveImage(Transform):
    """
    Save the image (in the form of torch tensor or numpy ndarray) and metadata dictionary into files.

    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the `input_image_name` is extracted from the provided metadata dictionary.
    If no metadata provided, a running index starting from 0 will be used as the filename prefix.

    Args:
        output_dir: output image directory.
            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_postfix: a string appended to all output file names, default to `trans`.
            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_ext: output file extension name.
            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        output_dtype: data type (if not None) for saving data. Defaults to ``np.float32``.
        resample: whether to resample image (if needed) before saving the data array,
            based on the ``"spatial_shape"`` (and ``"original_affine"``) from metadata.
        mode: This option is used when ``resample=True``. Defaults to ``"nearest"``.
            Depending on the writers, the possible options are

            - {``"bilinear"``, ``"nearest"``, ``"bicubic"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.
            Possible options are {``"zeros"``, ``"border"``, ``"reflection"``}
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (``uint8``) or [0, 65535] (``uint16``). Default is ``None`` (no scaling).
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if ``None``, use the data type of input data. To set the output data type, use ``output_dtype``.
        squeeze_end_dims: if ``True``, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D is also 1, it will be saved as (H,W). If ``False``,
            image will always be saved as (H,W,D,C).
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. It's used to compute ``input_file_rel_path``, the relative path to the file from
            ``data_root_dir`` to preserve folder structure when saving in case there are files in different
            folders with the same file names. For example, with the following inputs:

            - input_file_name: ``/foo/bar/test1/image.nii``
            - output_postfix: ``seg``
            - output_ext: ``.nii.gz``
            - output_dir: ``/output``
            - data_root_dir: ``/foo/bar``

            The output will be: ``/output/test1/image/image_seg.nii.gz``

            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        separate_folder: whether to save every file in a separate folder. For example: for the input filename
            ``image.nii``, postfix ``seg`` and ``folder_path`` ``output``, if ``separate_folder=True``, it will be
            saved as: ``output/image/image_seg.nii``, if ``False``, saving as ``output/image_seg.nii``.
            Default to ``True``.
            Handled by ``folder_layout`` instead, if ``folder_layout`` is not ``None``.
        print_log: whether to print logs when saving. Default to ``True``.
        output_format: an optional string of filename extension to specify the output image writer.
            see also: ``monai.data.image_writer.SUPPORTED_WRITERS``.
        writer: a customised ``monai.data.ImageWriter`` subclass to save data arrays.
            if ``None``, use the default writer from ``monai.data.image_writer`` according to ``output_ext``.
            if it's a string, it's treated as a class name or dotted path (such as ``"monai.data.ITKWriter"``);
            the supported built-in writer classes are ``"NibabelWriter"``, ``"ITKWriter"``, ``"PILWriter"``.
        channel_dim: the index of the channel dimension. Default to ``0``.
            ``None`` to indicate no channel dimension.
        output_name_formatter: a callable function (returning a kwargs dict) to format the output file name.
            If using a custom ``monai.data.FolderLayoutBase`` class in ``folder_layout``, consider providing
            your own formatter.
            see also: :py:func:`monai.data.folder_layout.default_name_formatter`.
        folder_layout: A customized ``monai.data.FolderLayoutBase`` subclass to define file naming schemes.
            if ``None``, uses the default ``FolderLayout``.
        savepath_in_metadict: if ``True``, adds a key ``"saved_to"`` to the metadata, which contains the path
            to where the input image has been saved.
    """

    def __init__(
        self,
        output_dir: PathLike = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        output_dtype: DtypeLike | None = np.float32,
        resample: bool = False,
        mode: str = "nearest",
        padding_mode: str = GridSamplePadMode.BORDER,
        scale: int | None = None,
        dtype: DtypeLike = np.float64,
        squeeze_end_dims: bool = True,
        data_root_dir: PathLike = "",
        separate_folder: bool = True,
        print_log: bool = True,
        output_format: str = "",
        writer: type[image_writer.ImageWriter] | str | None = None,
        channel_dim: int | None = 0,
        output_name_formatter: Callable[[dict, Transform], dict] | None = None,
        folder_layout: FolderLayoutBase | None = None,
        savepath_in_metadict: bool = False,
    ) -> None:
        self.folder_layout: FolderLayoutBase
        if folder_layout is None:
            self.folder_layout = FolderLayout(
                output_dir=output_dir,
                postfix=output_postfix,
                extension=output_ext,
                parent=separate_folder,
                makedirs=True,
                data_root_dir=data_root_dir,
            )
        else:
            self.folder_layout = folder_layout

        self.fname_formatter: Callable
        if output_name_formatter is None:
            self.fname_formatter = default_name_formatter
        else:
            self.fname_formatter = output_name_formatter

        self.output_ext = output_ext.lower() or output_format.lower()
        self.output_ext = (
            f".{self.output_ext}" if self.output_ext and not self.output_ext.startswith(".") else self.output_ext
        )
        if isinstance(writer, str):
            writer_, has_built_in = optional_import("monai.data", name=f"{writer}")  # search built-in
            if not has_built_in:
                writer_ = locate(f"{writer}")  # search dotted path
            if writer_ is None:
                raise ValueError(f"writer {writer} not found")
            writer = writer_
        self.writers = image_writer.resolve_writer(self.output_ext) if writer is None else (writer,)
        self.writer_obj = None

        _output_dtype = output_dtype
        if self.output_ext == ".png" and _output_dtype not in (np.uint8, np.uint16, None):
            _output_dtype = np.uint8
        if self.output_ext == ".dcm" and _output_dtype not in (np.uint8, np.uint16, None):
            _output_dtype = np.uint8
        self.init_kwargs = {"output_dtype": _output_dtype, "scale": scale}
        self.data_kwargs = {"squeeze_end_dims": squeeze_end_dims, "channel_dim": channel_dim}
        self.meta_kwargs = {"resample": resample, "mode": mode, "padding_mode": padding_mode, "dtype": dtype}
        self.write_kwargs = {"verbose": print_log}
        self._data_index = 0
        self.savepath_in_metadict = savepath_in_metadict

    def set_options(self, init_kwargs=None, data_kwargs=None, meta_kwargs=None, write_kwargs=None):
        """
        Set the options for the underlying writer by updating the `self.*_kwargs` dictionaries.

        The arguments correspond to the following usage:

            - `writer = ImageWriter(**init_kwargs)`
            - `writer.set_data_array(array, **data_kwargs)`
            - `writer.set_metadata(meta_data, **meta_kwargs)`
            - `writer.write(filename, **write_kwargs)`

        """
        if init_kwargs is not None:
            self.init_kwargs.update(init_kwargs)
        if data_kwargs is not None:
            self.data_kwargs.update(data_kwargs)
        if meta_kwargs is not None:
            self.meta_kwargs.update(meta_kwargs)
        if write_kwargs is not None:
            self.write_kwargs.update(write_kwargs)
        return self

    def __call__(
        self, img: torch.Tensor | np.ndarray, meta_data: dict | None = None, filename: str | PathLike | None = None
    ):
        """
        Args:
            img: target data content that save into file. The image should be channel-first, shape: `[C,H,W,[D]]`.
            meta_data: key-value pairs of metadata corresponding to the data.
            filename: str or file-like object which to save img.
                If specified, will ignore `self.output_name_formatter` and `self.folder_layout`.
        """
        meta_data = img.meta if isinstance(img, MetaTensor) else meta_data
        if filename is not None:
            filename = f"{filename}{self.output_ext}"
        else:
            kw = self.fname_formatter(meta_data, self)
            filename = self.folder_layout.filename(**kw)

        if meta_data:
            meta_spatial_shape = ensure_tuple(meta_data.get("spatial_shape", ()))
            if len(meta_spatial_shape) >= len(img.shape):
                self.data_kwargs["channel_dim"] = None
            elif is_no_channel(self.data_kwargs.get("channel_dim")):
                warnings.warn(
                    f"data shape {img.shape} (with spatial shape {meta_spatial_shape}) "
                    f"but SaveImage `channel_dim` is set to {self.data_kwargs.get('channel_dim')} no channel."
                )

        err = []
        for writer_cls in self.writers:
            try:
                writer_obj = writer_cls(**self.init_kwargs)
                writer_obj.set_data_array(data_array=img, **self.data_kwargs)
                writer_obj.set_metadata(meta_dict=meta_data, **self.meta_kwargs)
                writer_obj.write(filename, **self.write_kwargs)
                self.writer_obj = writer_obj
            except Exception as e:
                err.append(traceback.format_exc())
                logging.getLogger(self.__class__.__name__).debug(e, exc_info=True)
                logging.getLogger(self.__class__.__name__).info(
                    f"{writer_cls.__class__.__name__}: unable to write {filename}.\n"
                )
            else:
                self._data_index += 1
                if self.savepath_in_metadict and meta_data is not None:
                    meta_data[MetaKeys.SAVED_TO] = filename
                return img
        msg = "\n".join([f"{e}" for e in err])
        raise RuntimeError(
            f"{self.__class__.__name__} cannot find a suitable writer for {filename}.\n"
            "    Please install the writer libraries, see also the installation instructions:\n"
            "    https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies.\n"
            f"   The current registered writers for {self.output_ext}: {self.writers}.\n{msg}"
        )


class WriteFileMapping(Transform):
    """
    Writes a JSON file that logs the mapping between input image paths and their corresponding output paths.
    This class uses FileLock to ensure safe writing to the JSON file in a multiprocess environment.

    Args:
        mapping_file_path (Path or str): Path to the JSON file where the mappings will be saved.
    """

    def __init__(self, mapping_file_path: Path | str = "mapping.json"):
        self.mapping_file_path = Path(mapping_file_path)

    def __call__(self, img: NdarrayOrTensor):
        """
        Args:
            img: The input image with metadata.
        """
        if isinstance(img, MetaTensor):
            meta_data = img.meta

        if MetaKeys.SAVED_TO not in meta_data:
            raise KeyError(
                "Missing 'saved_to' key in metadata. Check SaveImage argument 'savepath_in_metadict' is True."
            )

        input_path = meta_data[Key.FILENAME_OR_OBJ]
        output_path = meta_data[MetaKeys.SAVED_TO]
        log_data = {"input": input_path, "output": output_path}

        if has_filelock:
            with FileLock(str(self.mapping_file_path) + ".lock"):
                self._write_to_file(log_data)
        else:
            self._write_to_file(log_data)
        return img

    def _write_to_file(self, log_data):
        try:
            with self.mapping_file_path.open("r") as f:
                existing_log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_log_data = []
        existing_log_data.append(log_data)
        with self.mapping_file_path.open("w") as f:
            json.dump(existing_log_data, f, indent=4)
