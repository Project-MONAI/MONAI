# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union, Callable

import numpy as np
from ignite.engine import Events, Engine
import logging
from monai.data import NiftiSaver, PNGSaver


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
        interp_order: int = 0,
        mode: str = "constant",
        cval: Union[int, float] = 0,
        scale: bool = False,
        dtype: Optional[np.dtype] = None,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
    ):
        """
        Args:
            output_dir (str): output image directory.
            output_postfix (str): a string appended to all output file names.
            output_ext (str): output file extension name.
            resample (bool): whether to resample before saving the data array.
            interp_order (int): the order of the spline interpolation, default is 0.
                The order has to be in the range 0 - 5.
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
                this option is used when `resample = True`.
            mode (`reflect|constant|nearest|mirror|wrap`):
                The mode parameter determines how the input array is extended beyond its boundaries.
                this option is used when `resample = True`.
            cval (scalar): Value to fill past edges of input if mode is "constant". Default is 0.0.
                this option is used when `resample = True`.
            scale (bool): whether to scale data with 255 and convert to uint8 for data in range [0, 1].
                it's used for PNG format only.
            dtype (np.dtype, optional): convert the image data to save to this data type.
                If None, keep the original type of data. it's used for Nifti format only.
            batch_transform (Callable): a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform (Callable): a callable that is used to transform the
                ignite.engine.output into the form expected image data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name (str): identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        if output_ext in (".nii.gz", ".nii"):
            self.saver = NiftiSaver(output_dir, output_postfix, output_ext, resample, interp_order, mode, cval, dtype)
        elif output_ext == ".png":
            self.saver = PNGSaver(output_dir, output_postfix, output_ext, resample, interp_order, mode, cval, scale)
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = None if name is None else logging.getLogger(name)

    def attach(self, engine: Engine):
        if self.logger is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, engine):
        """
        This method assumes self.batch_transform will extract metadata from the input batch.
        output file datatype is determined from ``engine.state.output.dtype``.

        """
        meta_data = self.batch_transform(engine.state.batch)
        engine_output = self.output_transform(engine.state.output)
        self.saver.save_batch(engine_output, meta_data)
        self.logger.info("saved all the model outputs into files.")
