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

import logging
import warnings
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from monai.config import DtypeLike
from monai.transforms import SaveImage
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class SegmentationSaver:
    """
    Event handler triggered on completing every iteration to save the segmentation predictions into files.
    It can extract the input image meta data(filename, affine, original_shape, etc.) and resample the predictions
    based on the meta data.

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
            batch_transform: a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
                it can be used to extract the input image meta data: filename, affine, original_shape, etc.
            output_transform: a callable that is used to transform the
                ignite.engine.output into the form expected image data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
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
        )
        self.resample = resample
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
        engine_output = self.output_transform(engine.state.output)
        if isinstance(engine_output, (tuple, list)):
            # if a list of data in shape: [channel, H, W, [D]], save every item separately
            if self.resample:
                warnings.warn("if saving inverted data, please set `resample=False` as it's already resampled.")

            self._saver.save_batch = False
            for i, d in enumerate(engine_output):
                self._saver(d, {k: meta_data[k][i] for k in meta_data} if meta_data is not None else None)
        else:
            # if the data is in shape: [batch, channel, H, W, [D]]
            self._saver.save_batch = True
            self._saver(engine_output, meta_data)
        self.logger.info("saved all the model outputs into files.")
