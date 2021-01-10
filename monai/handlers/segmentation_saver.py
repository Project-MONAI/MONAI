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
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np

from monai.data import NiftiSaver, PNGSaver
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")


class SegmentationSaver:
    """
    Event handler triggered on completing every iteration to save the segmentation predictions into files.
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
        dtype: Optional[np.dtype] = np.float64,
        output_dtype: Optional[np.dtype] = np.float32,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names.
            output_ext: output file extension name.
            resample: whether to resample before saving the data array.
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
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``, it's used for Nifti format only.
            output_dtype: data type for saving data. Defaults to ``np.float32``, it's used for Nifti format only.
            batch_transform: a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform: a callable that is used to transform the
                ignite.engine.output into the form expected image data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.

        """
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
        self.saver.save_batch(engine_output, meta_data)
        self.logger.info("saved all the model outputs into files.")
