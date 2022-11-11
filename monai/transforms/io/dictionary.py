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
A collection of dictionary-based wrappers around the "vanilla" transforms for IO functions
defined in :py:class:`monai.transforms.io.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from pathlib import Path
from typing import Optional, Type, Union

import numpy as np

from monai.config import DtypeLike, KeysCollection
from monai.data import image_writer
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage, SaveImage
from monai.transforms.transform import MapTransform
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix

__all__ = ["LoadImaged", "LoadImageD", "LoadImageDict", "SaveImaged", "SaveImageD", "SaveImageDict"]

DEFAULT_POST_FIX = PostFix.meta()


class LoadImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LoadImage`,
    It can load both image data and metadata. When loading a list of files in one key,
    the arrays will be stacked and a new dimension will be added as the first dimension
    In this case, the metadata of the first image will be used to represent the stacked result.
    The affine transform of all the stacked images should be same.
    The output metadata field will be created as ``meta_keys`` or ``key_{meta_key_postfix}``.

    If reader is not specified, this class automatically chooses readers
    based on the supported suffixes and in the following order:

        - User-specified reader at runtime when calling this loader.
        - User-specified reader in the constructor of `LoadImage`.
        - Readers from the last to the first in the registered list.
        - Current default readers: (nii, nii.gz -> NibabelReader), (png, jpg, bmp -> PILReader),
          (npz, npy -> NumpyReader), (dcm, DICOM series and others -> ITKReader).

    Please note that for png, jpg, bmp, and other 2D formats, readers often swap axis 0 and 1 after
    loading the array because the `HW` definition for non-medical specific file formats is different
    from other common medical packages.

    Note:

        - If `reader` is specified, the loader will attempt to use the specified readers and the default supported
          readers. This might introduce overheads when handling the exceptions of trying the incompatible loaders.
          In this case, it is therefore recommended setting the most appropriate reader as
          the last item of the `reader` parameter.

    See also:

        - tutorial: https://github.com/Project-MONAI/tutorials/blob/master/modules/load_medical_images.ipynb

    """

    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: Optional[str] = None,
        prune_meta_sep: str = ".",
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            reader: reader to load image file and metadata
                - if `reader` is None, a default set of `SUPPORTED_READERS` will be used.
                - if `reader` is a string, it's treated as a class name or dotted path
                (such as ``"monai.data.ITKReader"``), the supported built-in reader classes are
                ``"ITKReader"``, ``"NibabelReader"``, ``"NumpyReader"``.
                a reader instance will be constructed with the `*args` and `**kwargs` parameters.
                - if `reader` is a reader class/instance, it will be registered to this loader accordingly.
            dtype: if not None, convert the loaded image data to this data type.
            meta_keys: explicitly indicate the key to store the corresponding metadata dictionary.
                the metadata is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to store the metadata of the nifti image,
                default is `meta_dict`. The metadata is a dictionary object.
                For example, load nifti file for `image`, store the metadata into `image_meta_dict`.
            overwriting: whether allow overwriting existing metadata of same key.
                default is False, which will raise exception if encountering existing key.
            image_only: if True return dictionary containing just only the image volumes, otherwise return
                dictionary containing image data array and header dict per input key.
            ensure_channel_first: if `True` and loaded both image array and metadata, automatically convert
                the image array shape to `channel first`. default to `False`.
            simple_keys: whether to remove redundant metadata keys, default to False for backward compatibility.
            prune_meta_pattern: combined with `prune_meta_sep`, a regular expression used to match and prune keys
                in the metadata (nested dictionary), default to None, no key deletion.
            prune_meta_sep: combined with `prune_meta_pattern`, used to match and prune keys
                in the metadata (nested dictionary). default is ".", see also :py:class:`monai.transforms.DeleteItemsd`.
                e.g. ``prune_meta_pattern=".*_code$", prune_meta_sep=" "`` removes meta keys that ends with ``"_code"``.
            allow_missing_keys: don't raise exception if key is missing.
            args: additional parameters for reader if providing a reader name.
            kwargs: additional parameters for reader if providing a reader name.
        """
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(
            reader,
            image_only,
            dtype,
            ensure_channel_first,
            simple_keys,
            prune_meta_pattern,
            prune_meta_sep,
            *args,
            **kwargs,
        )
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        return d


class SaveImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SaveImage`.

    Note:
        Image should be channel-first shape: [C,H,W,[D]].
        If the data is a patch of an image, the patch index will be appended to the filename.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        meta_keys: explicitly indicate the key of the corresponding metadata dictionary.
            For example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            The metadata is a dictionary contains values such as filename, original_shape.
            This argument can be a sequence of string, map to the `keys`.
            If `None`, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if `meta_keys` is `None`, use `key_{meta_key_postfix}` to retrieve the metadict.
        output_dir: output image directory.
        output_postfix: a string appended to all output file names, default to `trans`.
        output_ext: output file extension name, available extensions: `.nii.gz`, `.nii`, `.png`.
        resample: whether to resample image (if needed) before saving the data array,
            based on the `spatial_shape` (and `original_affine`) from metadata.
        mode: This option is used when ``resample=True``. Defaults to ``"nearest"``.
            Depending on the writers, the possible options are:

            - {``"bilinear"``, ``"nearest"``, ``"bicubic"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}.
              See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.
            Possible options are {``"zeros"``, ``"border"``, ``"reflection"``}
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (uint8) or [0, 65535] (uint16). Default is `None` (no scaling).
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if None, use the data type of input data. To set the output data type, use `output_dtype`.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
        allow_missing_keys: don't raise exception if key is missing.
        squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D is also 1, it will be saved as (H,W). If `false`,
            image will always be saved as (H,W,D,C).
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. It's used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names. For example, with the following inputs:

            - input_file_name: `/foo/bar/test1/image.nii`
            - output_postfix: `seg`
            - output_ext: `.nii.gz`
            - output_dir: `/output`
            - data_root_dir: `/foo/bar`

            The output will be: /output/test1/image/image_seg.nii.gz

        separate_folder: whether to save every file in a separate folder. For example: for the input filename
            `image.nii`, postfix `seg` and folder_path `output`, if `separate_folder=True`, it will be saved as:
            `output/image/image_seg.nii`, if `False`, saving as `output/image_seg.nii`. Default to `True`.
        print_log: whether to print logs when saving. Default to `True`.
        output_format: an optional string to specify the output image writer.
            see also: `monai.data.image_writer.SUPPORTED_WRITERS`.
        writer: a customised `monai.data.ImageWriter` subclass to save data arrays.
            if `None`, use the default writer from `monai.data.image_writer` according to `output_ext`.
            if it's a string, it's treated as a class name or dotted path;
            the supported built-in writer classes are ``"NibabelWriter"``, ``"ITKWriter"``, ``"PILWriter"``.

    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        output_dir: Union[Path, str] = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: str = "nearest",
        padding_mode: str = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: DtypeLike = np.float64,
        output_dtype: Optional[DtypeLike] = np.float32,
        allow_missing_keys: bool = False,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        separate_folder: bool = True,
        print_log: bool = True,
        output_format: str = "",
        writer: Union[Type[image_writer.ImageWriter], str, None] = None,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.saver = SaveImage(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=output_ext,
            resample=resample,
            mode=mode,
            padding_mode=padding_mode,
            scale=scale,
            dtype=dtype,
            output_dtype=output_dtype,
            squeeze_end_dims=squeeze_end_dims,
            data_root_dir=data_root_dir,
            separate_folder=separate_folder,
            print_log=print_log,
            output_format=output_format,
            writer=writer,
        )

    def set_options(self, init_kwargs=None, data_kwargs=None, meta_kwargs=None, write_kwargs=None):
        self.saver.set_options(init_kwargs, data_kwargs, meta_kwargs, write_kwargs)

    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if meta_key is None and meta_key_postfix is not None:
                meta_key = f"{key}_{meta_key_postfix}"
            meta_data = d.get(meta_key) if meta_key is not None else None
            self.saver(img=d[key], meta_data=meta_data)
        return d


LoadImageD = LoadImageDict = LoadImaged
SaveImageD = SaveImageDict = SaveImaged
