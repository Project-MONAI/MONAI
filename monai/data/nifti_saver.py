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

from typing import Dict, Optional, Union

import numpy as np
import torch

from monai.config import DtypeLike, PathLike
from monai.data.nifti_writer import write_nifti
from monai.data.utils import create_file_basename
from monai.utils import GridSampleMode, GridSamplePadMode
from monai.utils import ImageMetaKey as Key
from monai.utils import deprecated


@deprecated(since="0.8", msg_suffix="use monai.transforms.SaveImage instead.")
class NiftiSaver:
    """
    Save the data as NIfTI file, it can support single data content or a batch of data.
    Typically, the data can be segmentation predictions, call `save` for single data
    or call `save_batch` to save a batch of data together.
    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the input image name is extracted from the provided metadata dictionary.
    If no metadata provided, use index from 0 as the filename prefix.

    Note: image should include channel dimension: [B],C,H,W,[D].

    .. deprecated:: 0.8
        Use :py:class:`monai.transforms.SaveImage` instead.

    """

    def __init__(
        self,
        output_dir: PathLike = "./",
        output_postfix: str = "seg",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
        squeeze_end_dims: bool = True,
        data_root_dir: PathLike = "",
        separate_folder: bool = True,
        print_log: bool = True,
    ) -> None:
        """
        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names.
            output_ext: output file extension name.
            resample: whether to convert the data array to it's original coordinate system
                based on `original_affine` in the `meta_data`.
            mode: {``"bilinear"``, ``"nearest"``}
                This option is used when ``resample = True``.
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                This option is used when ``resample = True``.
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data.
            output_dtype: data type for saving data. Defaults to ``np.float32``.
            squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
                then if C==1, it will be saved as (H,W,D). If D also ==1, it will be saved as (H,W). If false,
                image will always be saved as (H,W,D,C).
            data_root_dir: if not empty, it specifies the beginning parts of the input file's
                absolute path. it's used to compute `input_file_rel_path`, the relative path to the file from
                `data_root_dir` to preserve folder structure when saving in case there are files in different
                folders with the same file names. for example:
                input_file_name: /foo/bar/test1/image.nii,
                postfix: seg
                output_ext: nii.gz
                output_dir: /output,
                data_root_dir: /foo/bar,
                output will be: /output/test1/image/image_seg.nii.gz
            separate_folder: whether to save every file in a separate folder, for example: if input filename is
                `image.nii`, postfix is `seg` and folder_path is `output`, if `True`, save as:
                `output/image/image_seg.nii`, if `False`, save as `output/image_seg.nii`. default to `True`.
            print_log: whether to print log about the saved NIfTI file path, etc. default to `True`.

        """
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self.resample = resample
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.align_corners = align_corners
        self.dtype = dtype
        self.output_dtype = output_dtype
        self._data_index = 0
        self.squeeze_end_dims = squeeze_end_dims
        self.data_root_dir = data_root_dir
        self.separate_folder = separate_folder
        self.print_log = print_log

    def save(self, data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """
        Save data into a NIfTI file.
        The meta_data could optionally have the following keys:

            - ``'filename_or_obj'`` -- for output file name creation, corresponding to filename or object.
            - ``'original_affine'`` -- for data orientation handling, defaulting to an identity matrix.
            - ``'affine'`` -- for data output affine, defaulting to an identity matrix.
            - ``'spatial_shape'`` -- for data output shape.
            - ``'patch_index'`` -- if the data is a patch of big image, append the patch index to filename.

        When meta_data is specified and `resample=True`, the saver will try to resample batch data from the space
        defined by "affine" to the space defined by "original_affine".

        If meta_data is None, use the default index (starting from 0) as the filename.

        Args:
            data: target data content that to be saved as a NIfTI format file.
                Assuming the data shape starts with a channel dimension and followed by spatial dimensions.
            meta_data: the metadata information corresponding to the data.

        See Also
            :py:meth:`monai.data.nifti_writer.write_nifti`
        """
        filename = meta_data[Key.FILENAME_OR_OBJ] if meta_data else str(self._data_index)
        self._data_index += 1
        original_affine = meta_data.get("original_affine", None) if meta_data and self.resample else None
        affine = meta_data.get("affine", None) if meta_data else None
        spatial_shape = meta_data.get("spatial_shape", None) if meta_data else None
        patch_index = meta_data.get(Key.PATCH_INDEX, None) if meta_data else None

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        path = create_file_basename(
            postfix=self.output_postfix,
            input_file_name=filename,
            folder_path=self.output_dir,
            data_root_dir=self.data_root_dir,
            separate_folder=self.separate_folder,
            patch_index=patch_index,
        )
        path = f"{path}{self.output_ext}"
        # change data shape to be (channel, h, w, d)
        while len(data.shape) < 4:
            data = np.expand_dims(data, -1)
        # change data to "channel last" format and write to NIfTI format file
        data = np.moveaxis(np.asarray(data), 0, -1)

        # if desired, remove trailing singleton dimensions
        if self.squeeze_end_dims:
            while data.shape[-1] == 1:
                data = np.squeeze(data, -1)

        write_nifti(
            data,
            file_name=path,
            affine=affine,
            target_affine=original_affine,
            resample=True,
            output_spatial_shape=spatial_shape,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            dtype=self.dtype,
            output_dtype=self.output_dtype,
        )

        if self.print_log:
            print(f"file written: {path}.")

    def save_batch(self, batch_data: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None) -> None:
        """
        Save a batch of data into NIfTI format files.

        Spatially it supports up to three dimensions, that is, H, HW, HWD for
        1D, 2D, 3D respectively (with resampling supports for 2D and 3D only).

        When saving multiple time steps or multiple channels `batch_data`,
        time and/or modality axes should be appended after the batch dimensions.
        For example, the shape of a batch of 2D eight-class
        segmentation probabilities to be saved could be `(batch, 8, 64, 64)`;
        in this case each item in the batch will be saved as (64, 64, 1, 8)
        NIfTI file (the third dimension is reserved as a spatial dimension).

        Args:
            batch_data: target batch data content that save into NIfTI format.
            meta_data: every key-value in the meta_data is corresponding to a batch of data.

        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(data=data, meta_data={k: meta_data[k][i] for k in meta_data} if meta_data is not None else None)
