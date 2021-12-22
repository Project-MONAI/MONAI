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

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from monai.config import DtypeLike, IgniteInfo
from monai.data import decollate_batch
from monai.transforms import SaveImage
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode, deprecated, min_version, optional_import

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")


@deprecated(since="0.6.0", removed="0.9.0", msg_suffix="Please consider using `SaveImage[d]` transform instead.")
class SegmentationSaver:
    """
    Event handler triggered on completing every iteration to save the segmentation predictions into files.
    It can extract the input image meta data(filename, affine, original_shape, etc.) and resample the predictions
    based on the meta data.
    The name of saved file will be `{input_image_name}_{output_postfix}{output_ext}`,
    where the input image name is extracted from the meta data dictionary. If no meta data provided,
    use index from 0 as the filename prefix.
    The predictions can be PyTorch Tensor with [B, C, H, W, [D]] shape or a list of Tensor without batch dim.

    .. deprecated:: 0.6.0
        Use :class:`monai.transforms.SaveImage` or :class:`monai.transforms.SaveImaged` instead.

    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
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
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names, default to `seg`.
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
                It's used for PNG format only.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data.
                It's used for Nifti format only.
            output_dtype: data type for saving data. Defaults to ``np.float32``, it's used for Nifti format only.
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
            batch_transform: a callable that is used to extract the `meta_data` dictionary of the input images
                from `ignite.engine.state.batch`. the purpose is to extract necessary information from the meta data:
                filename, affine, original_shape, etc.
                `engine.state` and `batch_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            output_transform: a callable that is used to extract the model prediction data from
                `ignite.engine.state.output`. the first dimension of its output will be treated as the batch dimension.
                each item in the batch will be saved individually.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self._saver = SaveImage(
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
        )
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = logging.getLogger(name)
        self._name = name

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.
        Output file datatype is determined from ``engine.state.output.dtype``.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        meta_data = self.batch_transform(engine.state.batch)
        if isinstance(meta_data, dict):
            # decollate the `dictionary of list` to `list of dictionaries`
            meta_data = decollate_batch(meta_data)
        engine_output = self.output_transform(engine.state.output)
        for m, o in zip(meta_data, engine_output):
            self._saver(o, m)
        self.logger.info("model outputs saved into files.")
